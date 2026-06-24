document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("table").forEach(function (table) {
      // Only apply if it has a thead and more than one row
      if (table.querySelector("thead") && table.querySelectorAll("tbody tr").length > 1) {
        new Tablesort(table);
      }
    });
});
  