
# Generated: 2025-08-27T17:08:05.566854
# Source Brief: brief_01432.md
# Brief Index: 1432

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to plant. Shift to cycle plant type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate a thriving garden. Plant diverse flora, manage resources, and grow your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 12, 8
        self.CELL_SIZE = 40
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_W * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_H * self.CELL_SIZE) - 10

        self.MAX_STEPS = 1000
        self.WIN_SCORE = 500

        # --- Colors ---
        self.COLOR_BG = (35, 30, 40)
        self.COLOR_GRID = (60, 55, 65)
        self.COLOR_CURSOR = (220, 220, 255)
        self.COLOR_WATER = (80, 120, 255)
        self.COLOR_NUTRIENTS = (200, 180, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_DEAD_PLANT = (90, 70, 50)

        # --- Plant Definitions ---
        self.PLANT_TYPES = [
            {
                "name": "Vibra-Tulip",
                "color": (255, 80, 120),
                "cost": {"water": 30, "nutrients": 15},
                "consumption": {"water": 2, "nutrients": 1},
                "growth_interval": 20,
                "max_growth": 4,
                "score_per_growth": 15,
                "mature_score_rate": 2,
            },
            {
                "name": "Giga-Fern",
                "color": (80, 255, 150),
                "cost": {"water": 15, "nutrients": 30},
                "consumption": {"water": 1, "nutrients": 2},
                "growth_interval": 35,
                "max_growth": 4,
                "score_per_growth": 25,
                "mature_score_rate": 1,
            },
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 24, bold=True)

        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.cell_resources = None
        self.cursor_pos = None
        self.selected_plant_type = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.resource_replenish_rate = None
        
        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[None for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        self.cell_resources = [
            [{"water": 100.0, "nutrients": 100.0} for _ in range(self.GRID_W)]
            for _ in range(self.GRID_H)
        ]
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_plant_type = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.resource_replenish_rate = 1.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0.0

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # 1. Handle Player Actions
        self._handle_movement(movement)
        if shift_pressed:
            self.selected_plant_type = (self.selected_plant_type + 1) % len(self.PLANT_TYPES)
            # Small feedback for cycling
            self._create_particles(self.cursor_pos, self.PLANT_TYPES[self.selected_plant_type]['color'], 3, 5)

        if space_pressed:
            reward = self._handle_planting()
            step_reward += reward

        # 2. Update Game State
        plant_update_rewards, num_plants = self._update_plants()
        step_reward += plant_update_rewards

        self._update_resources()
        self._update_particles()

        # 3. Check Termination Conditions
        terminated = False
        if self.score >= self.WIN_SCORE:
            step_reward += 100.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        elif num_plants == 0 and not self._can_afford_any_plant():
            step_reward -= 100.0
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

    def _handle_planting(self):
        cx, cy = self.cursor_pos
        if self.grid[cy][cx] is not None:
            return -1.0  # Penalty for trying to plant on an occupied cell

        plant_info = self.PLANT_TYPES[self.selected_plant_type]
        cost = plant_info["cost"]
        cell_res = self.cell_resources[cy][cx]

        if cell_res["water"] >= cost["water"] and cell_res["nutrients"] >= cost["nutrients"]:
            cell_res["water"] -= cost["water"]
            cell_res["nutrients"] -= cost["nutrients"]
            self.grid[cy][cx] = {
                "type_id": self.selected_plant_type,
                "age": 0,
                "growth_stage": 0,
            }
            # sfx: planting sound
            self._create_particles(self.cursor_pos, (255, 255, 255), 15, 10)
            return 5.0
        else:
            # sfx: error/buzz sound
            self._create_particles(self.cursor_pos, (255, 0, 0), 5, 5)
            return -2.0  # Penalty for trying to plant without resources

    def _update_plants(self):
        total_reward = 0.0
        num_plants = 0
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                plant = self.grid[y][x]
                if plant is None:
                    continue
                
                num_plants += 1
                plant_info = self.PLANT_TYPES[plant["type_id"]]
                cell_res = self.cell_resources[y][x]
                consumption = plant_info["consumption"]

                # Consume resources
                cell_res["water"] -= consumption["water"]
                cell_res["nutrients"] -= consumption["nutrients"]
                total_reward -= (consumption["water"] + consumption["nutrients"]) * 0.1

                # Check for death
                if cell_res["water"] < 0 or cell_res["nutrients"] < 0:
                    self.grid[y][x] = None
                    total_reward -= 2.0
                    # sfx: plant wilting/dying sound
                    self._create_particles([x, y], self.COLOR_DEAD_PLANT, 20, 15)
                    continue

                # Age and Grow
                plant["age"] += 1
                if plant["growth_stage"] < plant_info["max_growth"]:
                    if plant["age"] % plant_info["growth_interval"] == 0:
                        plant["growth_stage"] += 1
                        total_reward += 1.0
                        self.score += plant_info["score_per_growth"]
                        # sfx: growth chime
                        self._create_particles([x, y], plant_info['color'], 5, 8, 0.5)

                # Mature plant scoring
                if plant["growth_stage"] == plant_info["max_growth"]:
                    self.score += plant_info["mature_score_rate"]

        return total_reward, num_plants

    def _update_resources(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.resource_replenish_rate = max(0.2, self.resource_replenish_rate - 0.2)

        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                res = self.cell_resources[y][x]
                res["water"] = min(100.0, res["water"] + self.resource_replenish_rate)
                res["nutrients"] = min(100.0, res["nutrients"] + self.resource_replenish_rate)

    def _can_afford_any_plant(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                for plant_type in self.PLANT_TYPES:
                    cost = plant_type["cost"]
                    res = self.cell_resources[y][x]
                    if res["water"] >= cost["water"] and res["nutrients"] >= cost["nutrients"]:
                        return True
        return False

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] *= 0.95

    def _create_particles(self, grid_pos, color, count, life, speed=1.0):
        cx, cy = self._grid_to_pixel(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 2.0) * speed
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * vel_mag
            self.particles.append({
                "pos": pygame.Vector2(cx, cy),
                "vel": vel,
                "life": self.np_random.integers(life // 2, life),
                "radius": self.np_random.uniform(2, 5),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, x, y):
        return (
            self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2,
        )

    def _render_game(self):
        # Draw grid cells with resource overlays
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                px, py = self._grid_to_pixel(x, y)
                rect = pygame.Rect(px - self.CELL_SIZE // 2, py - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)

                # Resource visualization
                res = self.cell_resources[y][x]
                water_alpha = int(res["water"] / 100 * 60)
                nutrient_alpha = int(res["nutrients"] / 100 * 60)

                if water_alpha > 0:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    s.fill((*self.COLOR_WATER, water_alpha))
                    self.screen.blit(s, rect.topleft)

                if nutrient_alpha > 0:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    s.fill((*self.COLOR_NUTRIENTS, nutrient_alpha))
                    self.screen.blit(s, rect.topleft)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw plants
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.grid[y][x] is not None:
                    self._draw_plant(x, y)
        
        # Draw particles
        for p in self.particles:
            if p["radius"] > 0.5:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p["color"])

        # Draw cursor
        cx, cy = self.cursor_pos
        px, py = self._grid_to_pixel(cx, cy)
        rect = pygame.Rect(px - self.CELL_SIZE // 2, py - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)

    def _draw_plant(self, x, y):
        plant = self.grid[y][x]
        plant_info = self.PLANT_TYPES[plant["type_id"]]
        color = plant_info["color"]
        px, py = self._grid_to_pixel(x, y)
        stage = plant["growth_stage"]

        if plant["type_id"] == 0: # Tulip
            if stage == 0: # Seed
                pygame.gfxdraw.filled_circle(self.screen, px, py, 3, self.COLOR_DEAD_PLANT)
            else:
                stem_height = min(self.CELL_SIZE // 2 - 5, 5 + stage * 4)
                pygame.draw.line(self.screen, (0, 150, 50), (px, py + 5), (px, py - stem_height), 3)
                if stage >= 2: # Bud
                    bud_size = 2 + (stage - 1) * 2
                    pygame.gfxdraw.filled_circle(self.screen, px, int(py - stem_height), bud_size, color)
                if stage >= 3: # Flower
                     pygame.gfxdraw.filled_circle(self.screen, px-4, int(py - stem_height - 2), 4, color)
                     pygame.gfxdraw.filled_circle(self.screen, px+4, int(py - stem_height - 2), 4, color)
                if stage == 4: # Mature Glow
                    glow_color = (*color, 50)
                    pygame.gfxdraw.filled_circle(self.screen, px, int(py - stem_height), 12, glow_color)
        
        elif plant["type_id"] == 1: # Fern
            if stage == 0: # Seed
                pygame.gfxdraw.filled_circle(self.screen, px, py, 3, self.COLOR_DEAD_PLANT)
            else:
                for i in range(stage * 2):
                    angle = i * (math.pi / (stage * 2)) - math.pi / 2
                    length = 5 + i * 1.5
                    end_x = px + length * math.cos(angle)
                    end_y = py + length * math.sin(angle)
                    pygame.draw.line(self.screen, color, (px, py), (int(end_x), int(end_y)), 2)
                if stage == 4: # Mature Glow
                    glow_color = (*color, 50)
                    pygame.gfxdraw.filled_circle(self.screen, px, py, 18, glow_color)

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_l.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))
        steps_text = self.font_m.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (15, 45))

        # Selected Plant Info
        plant_info = self.PLANT_TYPES[self.selected_plant_type]
        name_text = self.font_m.render(plant_info["name"], True, plant_info["color"])
        self.screen.blit(name_text, (self.WIDTH - 160, self.HEIGHT - 65))
        
        cost_w_text = self.font_s.render(f"WATER: {plant_info['cost']['water']}", True, self.COLOR_WATER)
        self.screen.blit(cost_w_text, (self.WIDTH - 160, self.HEIGHT - 45))
        cost_n_text = self.font_s.render(f"NUTR.: {plant_info['cost']['nutrients']}", True, self.COLOR_NUTRIENTS)
        self.screen.blit(cost_n_text, (self.WIDTH - 160, self.HEIGHT - 28))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play ---
    # This allows a human to play the game.
    # The agent's action is determined by keyboard inputs.
    
    # Mapping from Pygame keys to action components
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT: [0, 0, 1],
        pygame.K_RSHIFT: [0, 0, 1],
    }
    
    # Create a window to display the game
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Garden Cultivator")
    clock = pygame.time.Clock()

    print(GameEnv.user_guide)
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        took_action = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                
                # On keydown, set the corresponding action part and trigger a step
                if event.key in key_to_action:
                    act = key_to_action[event.key]
                    action[0] = act[0] if act[0] != 0 else action[0]
                    action[1] = act[1] if act[1] != 0 else action[1]
                    action[2] = act[2] if act[2] != 0 else action[2]
                    took_action = True
        
        # Since this is a turn-based env, we only step when an action is taken.
        if took_action:
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")
            if terminated:
                print("--- GAME OVER ---")
                if info['score'] >= env.WIN_SCORE:
                    print("YOU WIN!")
                else:
                    print("YOU LOSE!")
                # Game is over, will not step again until reset (press 'r')
                
        # Render the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    env.close()