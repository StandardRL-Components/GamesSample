from gymnasium.spaces import MultiDiscrete
import os
import pygame


import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math

# Set the SDL video driver to dummy to run Pygame headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a Mesopotamian trading game.

    The player controls a caravan, traveling between cities to craft goods
    from local resources and sell them for profit. The goal is to accumulate
    the most wealth within a time limit.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right
    - `actions[1]` (Space): 0=Released, 1=Held. Used to cycle crafting options.
    - `actions[2]` (Shift): 0=Released, 1=Held. Used to confirm a craft/trade action.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Reward Structure:**
    - Small negative reward per step to encourage efficiency.
    - Positive reward proportional to wealth gained from trades.
    - Bonus rewards for visiting multiple cities and reaching wealth milestones.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- FIX: Add required class attributes ---
    game_description = "Trade goods between ancient Mesopotamian cities to accumulate wealth within a time limit."
    user_guide = "Use arrow keys (↑↓←→) to travel. In a city, press SPACE to cycle craft options and SHIFT to craft/sell."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 1000
    PLAYER_SPEED = 8.0
    
    # Colors
    COLOR_BG = (210, 180, 140) # Parchment
    COLOR_ROUTE = (139, 69, 19) # Saddle Brown
    COLOR_PLAYER = (0, 120, 255)
    COLOR_PLAYER_GLOW = (100, 180, 255)
    COLOR_TEXT = (50, 30, 20)
    COLOR_TEXT_HEADER = (90, 60, 40)
    COLOR_UI_BG = (245, 222, 179, 180) # Wheat with alpha
    COLOR_GREEN = (40, 140, 40)
    COLOR_RED = (180, 40, 40)
    COLOR_HIGHLIGHT = (255, 215, 0) # Gold
    COLOR_DISABLED = (128, 128, 128)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("georgia", 18)
        self.font_header = pygame.font.SysFont("georgia", 24, bold=True)
        self.font_small = pygame.font.SysFont("georgia", 14)

        # --- Game Data ---
        self._define_game_data()

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_target_pos = np.array([0.0, 0.0])
        self.cities_data = []
        self.particles = []
        self.current_city_idx = -1
        self.crafting_selection_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.visited_cities = set()
        self.wealth_milestone_rewarded = False
        self.resource_replenishment_rate = 0.05

    def _define_game_data(self):
        """Initializes static data for cities, resources, and crafting."""
        self.CITIES = [
            {"name": "Ur", "pos": (100, 320)},
            {"name": "Uruk", "pos": (180, 250)},
            {"name": "Babylon", "pos": (280, 180)},
            {"name": "Nineveh", "pos": (400, 100)},
            {"name": "Ashur", "pos": (500, 160)},
        ]
        self.RESOURCES = ["Clay", "Reeds", "Grain", "Lumber"]
        self.CRAFTING_RECIPES = {
            "Pottery": {"cost": {"Clay": 2}, "value": 15},
            "Basket": {"cost": {"Reeds": 3}, "value": 18},
            "Beer": {"cost": {"Grain": 5}, "value": 25},
            "Furniture": {"cost": {"Lumber": 4, "Reeds": 1}, "value": 40},
            "Tablet": {"cost": {"Clay": 1, "Reeds": 1}, "value": 10},
        }
        self.crafting_options = list(self.CRAFTING_RECIPES.keys())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0  # Represents player's wealth
        self.game_over = False
        
        # Player state
        start_city_idx = self.np_random.integers(len(self.CITIES))
        start_city = self.CITIES[start_city_idx]
        self.player_pos = np.array(start_city["pos"], dtype=float)
        self.player_target_pos = np.array(start_city["pos"], dtype=float)
        
        # Game world state
        self.cities_data = self._initialize_cities()
        self.particles = []
        self.current_city_idx = -1
        self.crafting_selection_idx = 0
        
        # Control state
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True
        
        # Reward tracking
        self.visited_cities = set()
        self.wealth_milestone_rewarded = False
        self.resource_replenishment_rate = 0.05

        return self._get_observation(), self._get_info()

    def _initialize_cities(self):
        """Sets initial resources and market prices for all cities."""
        cities_data = []
        for i, city_template in enumerate(self.CITIES):
            city_data = {
                "name": city_template["name"],
                "pos": np.array(city_template["pos"]),
                "resources": {res: self.np_random.integers(80, 101) for res in self.RESOURCES},
                "market_prices": {
                    item: int(recipe["value"] * self.np_random.uniform(0.8, 1.2))
                    for item, recipe in self.CRAFTING_RECIPES.items()
                }
            }
            cities_data.append(city_data)
        return cities_data

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Time penalty

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # --- Game Logic ---
        self._update_world_state()

        if self.current_city_idx != -1:
            # Player is in a city
            if space_press:
                self.crafting_selection_idx = (self.crafting_selection_idx + 1) % len(self.crafting_options)
            if shift_press:
                trade_reward = self._handle_craft_and_trade()
                reward += trade_reward
        else:
            # Player is on the map
            self._handle_movement(movement)
        
        self._update_player_position()
        self._check_for_city_entry()
        
        # --- Reward Calculation ---
        if not self.wealth_milestone_rewarded and self.score >= 500:
            reward += 50
            self.wealth_milestone_rewarded = True
        
        # --- Termination ---
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_world_state(self):
        """Update dynamic elements of the world each step."""
        if self.steps > 0 and self.steps % 200 == 0:
            self.resource_replenishment_rate = max(0.01, self.resource_replenishment_rate - 0.01)

        if self.steps % 10 == 0:
            for city in self.cities_data:
                for res in city["resources"]:
                    if self.np_random.random() < self.resource_replenishment_rate:
                        city["resources"][res] = min(100, city["resources"][res] + 1)
        
        if self.steps % 30 == 0:
            for city in self.cities_data:
                for item in city["market_prices"]:
                    change = self.np_random.uniform(-0.05, 0.05)
                    base_value = self.CRAFTING_RECIPES[item]["value"]
                    city["market_prices"][item] = int(max(base_value * 0.5, city["market_prices"][item] * (1 + change)))

    def _handle_movement(self, movement_action):
        """Updates player's target position based on movement action."""
        move_vec = np.array([0.0, 0.0])
        if movement_action == 1: move_vec[1] = -1 # Up
        elif movement_action == 2: move_vec[1] = 1  # Down
        elif movement_action == 3: move_vec[0] = -1 # Left
        elif movement_action == 4: move_vec[0] = 1  # Right
        
        if np.any(move_vec):
            self.player_target_pos = self.player_pos + move_vec * self.PLAYER_SPEED * 5

    def _update_player_position(self):
        """Interpolates player position towards the target and handles boundaries."""
        direction = self.player_target_pos - self.player_pos
        distance = np.linalg.norm(direction)
        
        if distance > 1:
            self.player_pos += direction / distance * min(distance, self.PLAYER_SPEED)
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)
        self.player_target_pos[0] = np.clip(self.player_target_pos[0], 0, self.SCREEN_WIDTH)
        self.player_target_pos[1] = np.clip(self.player_target_pos[1], 0, self.SCREEN_HEIGHT)

    def _check_for_city_entry(self):
        """Checks if the player has entered a city's radius."""
        self.current_city_idx = -1
        for i, city in enumerate(self.cities_data):
            if np.linalg.norm(self.player_pos - city["pos"]) < 20:
                self.current_city_idx = i
                # --- FIX: Ensure player position remains float to prevent TypeError ---
                self.player_pos = city["pos"].copy().astype(float)
                self.player_target_pos = city["pos"].copy().astype(float)
                if i not in self.visited_cities:
                    self.visited_cities.add(i)
                    if len(self.visited_cities) > 1 and len(self.visited_cities) % 3 == 0:
                        self.score += 5
                        self._create_particles(self.player_pos, self.COLOR_HIGHLIGHT, 20, 5)
                break

    def _handle_craft_and_trade(self):
        """Handles the logic for crafting and immediately selling an item."""
        city = self.cities_data[self.current_city_idx]
        item_name = self.crafting_options[self.crafting_selection_idx]
        recipe = self.CRAFTING_RECIPES[item_name]
        
        can_craft = all(city["resources"][res] >= amount for res, amount in recipe["cost"].items())
        
        if can_craft:
            for resource, amount_needed in recipe["cost"].items():
                city["resources"][resource] -= amount_needed
            
            sale_price = city["market_prices"][item_name]
            self.score += sale_price
            
            self._create_particles(self.player_pos, self.COLOR_HIGHLIGHT, 30, sale_price / 2)
            
            return min(10, sale_price / 5)
        else:
            self._create_particles(self.player_pos, self.COLOR_RED, 10, 2)
            return 0

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "current_city": self.cities_data[self.current_city_idx]["name"] if self.current_city_idx != -1 else "On Map",
        }
        
    def _render_all(self):
        """Master rendering function."""
        self.screen.fill(self.COLOR_BG)
        self._render_routes()
        self._render_particles()
        self._render_cities()
        self._render_player()
        self._render_ui()

    def _render_routes(self):
        """Draws lines connecting all cities."""
        for i in range(len(self.CITIES)):
            for j in range(i + 1, len(self.CITIES)):
                pygame.draw.line(self.screen, self.COLOR_ROUTE, self.CITIES[i]["pos"], self.CITIES[j]["pos"], 2)

    def _render_cities(self):
        """Renders cities and their resource bars."""
        for i, city in enumerate(self.cities_data):
            pos = tuple(map(int, city["pos"]))
            is_current = i == self.current_city_idx
            
            color = self.COLOR_HIGHLIGHT if is_current else (255, 255, 224)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_ROUTE)
            
            name_surf = self.font_small.render(city["name"], True, self.COLOR_TEXT)
            self.screen.blit(name_surf, (pos[0] - name_surf.get_width() // 2, pos[1] + 18))
            
            res_colors = [(205, 92, 92), (34, 139, 34), (218, 165, 32), (160, 82, 45)]
            for r_idx, (res, amount) in enumerate(city["resources"].items()):
                bar_x = pos[0] - 10
                bar_y = pos[1] - 25 - r_idx * 4
                bar_width = int(amount / 100 * 20)
                pygame.draw.rect(self.screen, res_colors[r_idx], (bar_x, bar_y, bar_width, 3))

    def _render_player(self):
        """Renders the player's caravan with a glow effect."""
        pos = tuple(map(int, self.player_pos))
        glow_radius = int(12 + 4 * math.sin(self.steps * 0.1))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, (0, 80, 200))

    def _render_ui(self):
        """Renders all UI elements, including the main HUD and city interface."""
        score_text = f"Wealth: {self.score}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT_HEADER)
        self.screen.blit(score_surf, (10, 10))

        time_text = f"Day: {self.steps}/{self.MAX_STEPS}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT_HEADER)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))

        if self.current_city_idx != -1:
            self._render_city_ui()

    def _render_city_ui(self):
        """Renders the detailed interface when inside a city."""
        city = self.cities_data[self.current_city_idx]
        
        ui_surf = pygame.Surface((300, 200), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        
        city_name_surf = self.font_header.render(city["name"], True, self.COLOR_TEXT_HEADER)
        ui_surf.blit(city_name_surf, (150 - city_name_surf.get_width() // 2, 10))

        res_y = 50
        for res, amount in city["resources"].items():
            res_surf = self.font_small.render(f"{res}: {amount}", True, self.COLOR_TEXT)
            ui_surf.blit(res_surf, (20, res_y))
            res_y += 20

        selected_item = self.crafting_options[self.crafting_selection_idx]
        recipe = self.CRAFTING_RECIPES[selected_item]
        
        ui_surf.blit(self.font_main.render("Craft & Sell:", True, self.COLOR_TEXT), (160, 50))
        
        pygame.draw.rect(ui_surf, self.COLOR_HIGHLIGHT, (155, 75, 140, 25), border_radius=5)
        ui_surf.blit(self.font_main.render(selected_item, True, self.COLOR_TEXT), (160, 78))

        cost_y = 110
        can_craft = all(city["resources"][res] >= amount for res, amount in recipe["cost"].items())
        for res, amount_needed in recipe["cost"].items():
            has_enough = city["resources"][res] >= amount_needed
            color = self.COLOR_GREEN if has_enough else self.COLOR_RED
            cost_surf = self.font_small.render(f"- {amount_needed} {res}", True, color)
            ui_surf.blit(cost_surf, (160, cost_y))
            cost_y += 18
            
        profit = city["market_prices"][selected_item]
        profit_color = self.COLOR_DISABLED if not can_craft else self.COLOR_GREEN
        profit_surf = self.font_small.render(f"Profit: {profit}", True, profit_color)
        ui_surf.blit(profit_surf, (160, cost_y + 5))

        ui_surf.blit(self.font_small.render("[SPACE] to cycle", True, self.COLOR_TEXT), (10, 175))
        ui_surf.blit(self.font_small.render("[SHIFT] to confirm", True, self.COLOR_TEXT), (170, 175))
        
        self.screen.blit(ui_surf, (self.SCREEN_WIDTH // 2 - 150, self.SCREEN_HEIGHT // 2 - 100))

    def _create_particles(self, pos, color, count, speed_mult):
        """Creates a burst of particles."""
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": velocity,
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _render_particles(self):
        """Updates and renders all active particles."""
        remaining_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9
            p["life"] -= 1
            if p["life"] > 0:
                size = int(p["life"] / 10)
                if size > 0:
                    pygame.draw.circle(self.screen, p["color"], p["pos"].astype(int), size)
                remaining_particles.append(p)
        self.particles = remaining_particles

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override Pygame screen for display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Mesopotamian Trader")
    
    total_reward = 0
    
    # Game loop
    running = True
    while running and not done:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render to the display window
        pygame.display.flip()
        
        env.clock.tick(30)

    print(f"Episode finished. Total reward: {total_reward:.2f}, Final score: {info['score']}")
    env.close()