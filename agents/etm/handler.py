"""
ETM Dispatcher Handler — processes ETM_PRICE and ETM_REMAINS tasks.
Called by Dispatcher when a task of type etm_price/etm_remains arrives.

Returns combined price + stock data for each product.
"""
import structlog
from agents.dispatcher.task import Task
from agents.etm.client import get_etm_client

logger = structlog.get_logger()


async def handle_etm_price(task: Task) -> dict:
    """
    Handle ETM_PRICE task: get prices AND remains for list of product IDs.

    Expected task.payload:
        {
            "product_ids": ["9536092", "1037375", ...],
            "id_type": "etm"   # optional, default "etm"
        }

    Returns:
        {
            "products": [
                {
                    "gdscode": 9536092,
                    "price": 939.42,
                    "pricewnds": 1146.09,
                    "price_tarif": 1469.36,
                    "price_retail": 1468.46,
                    "remains": {
                        "unit": "шт",
                        "stores": [...],
                        "delivery_days": "4",
                        "total_stock": 2515
                    }
                },
                ...
            ]
        }
    """
    client = get_etm_client()
    product_ids = task.payload.get("product_ids", [])
    id_type = task.payload.get("id_type", "etm")

    if not product_ids:
        return {"error": "No product_ids in payload", "products": []}

    logger.info("etm_handler_price", ids=product_ids, type=id_type)

    # 1. Get prices (batch)
    prices = await client.get_prices(product_ids, id_type=id_type)

    # Index prices by gdscode
    price_map = {}
    for p in prices:
        code = str(p.get("gdscode", ""))
        price_map[code] = p

    # 2. Get remains for each product (one by one, API limitation)
    products = []
    for pid in product_ids:
        product = dict(price_map.get(pid, {"gdscode": pid}))

        # Fetch remains
        remains_raw = await client.get_remains(pid, id_type=id_type)

        if remains_raw and "error" not in remains_raw:
            stores = remains_raw.get("InfoStores", [])
            total_stock = sum(
                int(s.get("StoreQuantRem", 0))
                for s in stores
                if str(s.get("StoreQuantRem", "0")).isdigit()
            )

            # Simplify store info
            store_list = []
            for s in stores:
                qty = s.get("StoreQuantRem", 0)
                if int(qty) > 0:
                    name = s.get("StoreName", "").strip()
                    if not name:
                        stype = s.get("StoreType", "")
                        name = {"rc": "Региональный центр", "crs": "Логистический центр", "op": "Офис продаж"}.get(stype, "Склад")
                    store_list.append({
                        "name": name,
                        "type": s.get("StoreType", ""),
                        "quantity": int(qty),
                    })

            # Supplier stores
            supp_stores = remains_raw.get("InfoSuppStores", [])
            for ss in supp_stores:
                qty = ss.get("SuppStoreQuantRem", 0)
                try:
                    qty_int = int(qty)
                except (ValueError, TypeError):
                    qty_int = 0
                if qty_int > 0:
                    store_list.append({
                        "name": ss.get("SuppStoreName", "Склад производителя"),
                        "type": "supplier",
                        "quantity": qty_int,
                    })
                    total_stock += qty_int

            delivery = remains_raw.get("InforDeliveryTime", {})

            product["remains"] = {
                "unit": remains_raw.get("UnitName", ""),
                "stores": store_list,
                "total_stock": total_stock,
                "delivery_days": delivery.get("DeliveryTimeInPres", ""),
                "production_term": delivery.get("DeliveryProductionTerm", ""),
            }
        else:
            product["remains"] = {
                "unit": "",
                "stores": [],
                "total_stock": 0,
                "delivery_days": "",
                "error": remains_raw.get("error", "No data"),
            }

        products.append(product)

    logger.info("etm_handler_price_done", count=len(products))
    return {"products": products}


async def handle_etm_remains(task: Task) -> dict:
    """Handle ETM_REMAINS task for a single product."""
    client = get_etm_client()
    product_id = task.payload.get("product_id", "")
    id_type = task.payload.get("id_type", "etm")

    if not product_id:
        return {"error": "No product_id in payload"}

    remains = await client.get_remains(product_id, id_type=id_type)
    return remains
