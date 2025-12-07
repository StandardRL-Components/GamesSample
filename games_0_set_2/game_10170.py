import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:02:14.123698
# Source Brief: brief_00170.md
# Brief Index: 170
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import defaultdict

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Explore a vast cave system, mine valuable ores, and fend off dangerous creatures. "
        "Craft powerful gear to increase your score and survive as long as possible."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to mine ore from adjacent walls "
        "and hold shift to attack creatures."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    TILE_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_WALL = (40, 30, 50)
    COLOR_FLOOR = (60, 50, 70)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (20, 20, 30)
    COLOR_UI_BG = (30, 20, 40, 180)
    
    ORE_COLORS = {
        "Copper": (200, 100, 20),
        "Iron": (160, 170, 180),
        "Mithril": (80, 150, 255),
        "Adamantite": (220, 50, 150)
    }

    # Game Parameters
    PLAYER_SPEED = 5
    PLAYER_MAX_HEALTH = 100
    PLAYER_ATTACK_DAMAGE = 10
    PLAYER_ATTACK_COOLDOWN = 15 # steps
    PLAYER_MINE_COOLDOWN = 10 # steps
    
    CREATURE_BASE_HEALTH = 30
    CREATURE_BASE_DAMAGE = 5
    CREATURE_SIGHT_RANGE = 150
    CREATURE_ATTACK_RANGE = 25
    
    # Crafting
    CRAFTING_RECIPES = {
        "Copper Dagger": {"req": {"Copper": 5}, "score": 25, "unlock_score": 0},
        "Iron Pickaxe": {"req": {"Iron": 8}, "score": 60, "unlock_score": 100},
        "Mithril Mail": {"req": {"Mithril": 10, "Iron": 5}, "score": 200, "unlock_score": 300},
        "Adamantite Greatsword": {"req": {"Adamantite": 12, "Mithril": 8}, "score": 500, "unlock_score": 1000},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        self.render_mode = render_mode
        # self.reset() is called by the wrapper/runner
        # self.validate_implementation() is for debugging, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0.0
        
        self._generate_world()

        self.player = {
            "pos": pygame.Vector2(self.start_pos) * self.TILE_SIZE + pygame.Vector2(self.TILE_SIZE/2, self.TILE_SIZE/2),
            "health": self.PLAYER_MAX_HEALTH,
            "inventory": defaultdict(int),
            "last_move_dir": pygame.Vector2(1, 0),
            "attack_cooldown": 0,
            "mine_cooldown": 0
        }
        
        self.creatures = []
        self._spawn_creatures(3)
        
        self.particles = []
        self.crafted_items = set()
        self.unlocked_recipes = self._get_unlocked_recipes()
        self.score_bonus_10k_claimed = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        self._handle_player_actions(action)
        self._update_creatures()
        self._update_player_state()
        self._check_and_perform_crafting()
        self._update_particles()
        
        # Spawning and progression
        if self.steps > 0 and self.steps % 500 == 0:
            self._spawn_creatures(1, difficulty_multiplier=1 + (self.steps // 500) * 0.1)
        
        terminated = self.player["health"] <= 0 or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True

        if self.score >= 10000 and not self.score_bonus_10k_claimed:
            self.reward_this_step += 100
            self.score_bonus_10k_claimed = True

        return self._get_observation(), self.reward_this_step, terminated, truncated, self._get_info()

    def _generate_world(self):
        self.world_grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int) # 1 = wall
        self.ore_veins = {}

        # Random walk to carve caves
        x, y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.start_pos = (x, y)
        self.world_grid[x, y] = 0 # 0 = floor
        num_floors = self.GRID_WIDTH * self.GRID_HEIGHT // 3
        for _ in range(num_floors):
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            x, y = max(1, min(self.GRID_WIDTH - 2, x + dx)), max(1, min(self.GRID_HEIGHT - 2, y + dy))
            self.world_grid[x, y] = 0
        
        # Place ores
        ore_tiers = {
            "Copper": (0, 0.05),
            "Iron": (100, 0.04),
            "Mithril": (300, 0.03),
            "Adamantite": (1000, 0.02)
        }
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                if self.world_grid[x, y] == 1: # If it's a wall
                    is_adjacent_to_floor = False
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        if self.world_grid[x+dx, y+dy] == 0:
                            is_adjacent_to_floor = True
                            break
                    if is_adjacent_to_floor:
                        for ore_name, (unlock_score, probability) in ore_tiers.items():
                             if self.score >= unlock_score and self.np_random.random() < probability:
                                self.ore_veins[(x, y)] = ore_name
                                break

    def _spawn_creatures(self, num, difficulty_multiplier=1.0):
        for _ in range(num):
            while True:
                x = self.np_random.integers(1, self.GRID_WIDTH - 1)
                y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
                if self.world_grid[x, y] == 0:
                    pos = pygame.Vector2(x, y) * self.TILE_SIZE + pygame.Vector2(self.TILE_SIZE/2, self.TILE_SIZE/2)
                    if pos.distance_to(self.player["pos"]) > 100:
                        max_health = int(self.CREATURE_BASE_HEALTH * difficulty_multiplier)
                        self.creatures.append({
                            "pos": pos,
                            "health": max_health,
                            "max_health": max_health,
                            "damage": int(self.CREATURE_BASE_DAMAGE * difficulty_multiplier),
                            "attack_cooldown": 0,
                            "move_cooldown": 0,
                        })
                        break

    def _handle_player_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        if move_vec.length() > 0:
            self.player["last_move_dir"] = move_vec.normalize()
            self.reward_this_step -= 0.01 # Small cost for moving
        
        new_pos = self.player["pos"] + move_vec * self.PLAYER_SPEED
        grid_x, grid_y = int(new_pos.x / self.TILE_SIZE), int(new_pos.y / self.TILE_SIZE)
        if self.world_grid[grid_x, grid_y] == 0: # Is floor
            self.player["pos"] = new_pos
        
        # Attack
        if shift_held and self.player["attack_cooldown"] == 0:
            self.player["attack_cooldown"] = self.PLAYER_ATTACK_COOLDOWN
            # SFX: Sword swing
            attack_pos = self.player["pos"] + self.player["last_move_dir"] * self.TILE_SIZE
            self._create_particles(10, attack_pos, (255, 200, 200), 0.5, 8, 10) # Slash effect
            
            for creature in self.creatures[:]:
                if creature["pos"].distance_to(attack_pos) < self.TILE_SIZE:
                    creature["health"] -= self.PLAYER_ATTACK_DAMAGE
                    self._create_particles(5, creature["pos"], (255, 0, 0), 1, 5, 15) # Hit effect
                    if creature["health"] <= 0:
                        # SFX: Creature death
                        self.reward_this_step += 50
                        self.creatures.remove(creature)
                        # Drop loot
                        ore_type = random.choice(list(self.ORE_COLORS.keys()))
                        self.player["inventory"][ore_type] += 1
                    break

        # Mining
        if space_held and self.player["mine_cooldown"] == 0:
            mine_pos_grid = (int((self.player["pos"].x + self.player["last_move_dir"].x * self.TILE_SIZE) / self.TILE_SIZE),
                             int((self.player["pos"].y + self.player["last_move_dir"].y * self.TILE_SIZE) / self.TILE_SIZE))
            
            if mine_pos_grid in self.ore_veins:
                # SFX: Mining clink
                self.player["mine_cooldown"] = self.PLAYER_MINE_COOLDOWN
                ore_type = self.ore_veins.pop(mine_pos_grid)
                self.player["inventory"][ore_type] += 1
                self.reward_this_step += 1
                mine_world_pos = pygame.Vector2(mine_pos_grid) * self.TILE_SIZE + pygame.Vector2(self.TILE_SIZE/2, self.TILE_SIZE/2)
                self._create_particles(15, mine_world_pos, self.ORE_COLORS[ore_type], 0.8, 4, 20)

    def _update_player_state(self):
        if self.player["attack_cooldown"] > 0: self.player["attack_cooldown"] -= 1
        if self.player["mine_cooldown"] > 0: self.player["mine_cooldown"] -= 1
        self.player["health"] = min(self.PLAYER_MAX_HEALTH, self.player["health"])

    def _update_creatures(self):
        for creature in self.creatures:
            if creature["attack_cooldown"] > 0: creature["attack_cooldown"] -= 1
            if creature["move_cooldown"] > 0: creature["move_cooldown"] -= 1

            dist_to_player = creature["pos"].distance_to(self.player["pos"])
            if dist_to_player < self.CREATURE_SIGHT_RANGE:
                if dist_to_player < self.CREATURE_ATTACK_RANGE:
                    if creature["attack_cooldown"] == 0:
                        # SFX: Player hurt
                        self.player["health"] -= creature["damage"]
                        creature["attack_cooldown"] = 30 # steps
                        self._create_particles(10, self.player["pos"], (200, 0, 0), 1, 6, 10)
                elif creature["move_cooldown"] == 0:
                    direction = (self.player["pos"] - creature["pos"]).normalize()
                    new_pos = creature["pos"] + direction * (self.PLAYER_SPEED * 0.5)
                    grid_x, grid_y = int(new_pos.x / self.TILE_SIZE), int(new_pos.y / self.TILE_SIZE)
                    if self.world_grid[grid_x, grid_y] == 0:
                        creature["pos"] = new_pos
            elif creature["move_cooldown"] == 0: # Patrol
                move_vec = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize()
                new_pos = creature["pos"] + move_vec * (self.PLAYER_SPEED * 0.3)
                grid_x, grid_y = int(new_pos.x / self.TILE_SIZE), int(new_pos.y / self.TILE_SIZE)
                if self.world_grid[grid_x, grid_y] == 0:
                    creature["pos"] = new_pos
                creature["move_cooldown"] = 5

    def _check_and_perform_crafting(self):
        self.unlocked_recipes = self._get_unlocked_recipes()
        for name, data in self.CRAFTING_RECIPES.items():
            if name in self.unlocked_recipes and name not in self.crafted_items:
                can_craft = True
                for ore, amount in data["req"].items():
                    if self.player["inventory"][ore] < amount:
                        can_craft = False
                        break
                if can_craft:
                    # SFX: Crafting success
                    for ore, amount in data["req"].items():
                        self.player["inventory"][ore] -= amount
                    self.crafted_items.add(name)
                    self.score += data["score"]
                    self.reward_this_step += 10

    def _get_unlocked_recipes(self):
        return {name for name, data in self.CRAFTING_RECIPES.items() if self.score >= data["unlock_score"]}

    def _create_particles(self, count, pos, color, speed, size, lifespan):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed * self.np_random.uniform(0.5, 1.5)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifespan": lifespan, "max_life": lifespan, "color": color, "size": size})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player["health"], "inventory": dict(self.player["inventory"])}

    def _render_game(self):
        # Draw world grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.world_grid[x, y] == 1:
                    color = self.ORE_COLORS.get(self.ore_veins.get((x,y)), self.COLOR_WALL)
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_life"]))
            size = int(p["size"] * (p["lifespan"] / p["max_life"]))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), size, (*p["color"], alpha))

        # Draw creatures
        for creature in self.creatures:
            pygame.draw.circle(self.screen, (200, 20, 20), (int(creature["pos"].x), int(creature["pos"].y)), self.TILE_SIZE // 2)
            # Health bar
            health_ratio = creature["health"] / creature["max_health"]
            bar_w = int(self.TILE_SIZE * 1.2)
            bar_h = 5
            bar_x = creature["pos"].x - bar_w / 2
            bar_y = creature["pos"].y - self.TILE_SIZE
            pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, bar_y, bar_w * health_ratio, bar_h))
            
        # Draw player
        px, py = int(self.player["pos"].x), int(self.player["pos"].y)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.TILE_SIZE // 2 + 3, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.TILE_SIZE // 2 + 3, self.COLOR_PLAYER_GLOW)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (px, py), self.TILE_SIZE // 2)

    def _render_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, align="topleft"):
        text_surf = font.render(str(text), True, color)
        shadow_surf = font.render(str(text), True, shadow_color)
        text_rect = text_surf.get_rect()
        if align == "topleft": text_rect.topleft = pos
        elif align == "topright": text_rect.topright = pos
        elif align == "midtop": text_rect.midtop = pos
        
        shadow_pos = (text_rect.x + 2, text_rect.y + 2)
        self.screen.blit(shadow_surf, shadow_pos)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        ui_box_score = pygame.Rect(5, 5, 200, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_box_score, border_radius=5)
        self._render_text(f"Score: {self.score}", (15, 15), self.font_small)

        # Inventory
        inv_w = 220
        inv_h = 100
        ui_box_inv = pygame.Rect(self.SCREEN_WIDTH - inv_w - 5, 5, inv_w, inv_h)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_box_inv, border_radius=5)
        self._render_text("Inventory", (self.SCREEN_WIDTH - inv_w, 10), self.font_small)
        y_offset = 30
        for ore_type, count in self.player["inventory"].items():
            if count > 0:
                ore_color = self.ORE_COLORS[ore_type]
                pygame.draw.circle(self.screen, ore_color, (self.SCREEN_WIDTH - inv_w + 20, 10 + y_offset), 6)
                self._render_text(f"{ore_type}: {count}", (self.SCREEN_WIDTH - inv_w + 35, 5 + y_offset), self.font_small)
                y_offset += 20

        # Health Bar
        health_bar_width = 300
        health_bar_height = 20
        health_bar_pos_x = (self.SCREEN_WIDTH - health_bar_width) / 2
        health_bar_pos_y = self.SCREEN_HEIGHT - health_bar_height - 10
        health_ratio = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (health_bar_pos_x - 5, health_bar_pos_y - 5, health_bar_width + 10, health_bar_height + 10), border_radius=5)
        pygame.draw.rect(self.screen, (50,0,0), (health_bar_pos_x, health_bar_pos_y, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, (0, 200, 50), (health_bar_pos_x, health_bar_pos_y, health_bar_width * health_ratio, health_bar_height))
        self._render_text(f"{int(self.player['health'])} / {self.PLAYER_MAX_HEALTH}", (self.SCREEN_WIDTH/2, health_bar_pos_y + 2), self.font_small, align="midtop")

        # Game Over
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            self._render_text("GAME OVER", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 30), self.font_large, align="midtop")
            self._render_text(f"Final Score: {self.score}", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20), self.font_small, align="midtop")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not be run by the evaluator
    
    # The following code is intended for human play and visualization.
    # It sets up a pygame window and maps keyboard inputs to actions.
    
    # To run, you might need to unset the dummy video driver
    # depending on your OS and environment.
    # For example, on Linux, you might run:
    # SDL_VIDEODRIVER=x11 python your_script_name.py
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="human_playable")
    obs, info = env.reset()
    done = False
    
    # Override screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cavern Crafter")
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Action mapping
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render to the display window
        rendered_frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rendered_frame)
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for playability
        
    env.close()