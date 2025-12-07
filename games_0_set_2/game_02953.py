
# Generated: 2025-08-28T06:31:02.138706
# Source Brief: brief_02953.md
# Brief Index: 2953

        
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

    user_guide = (
        "Controls: ↑↓←→ to move. Space to attack. No-op or Shift to defend."
    )

    game_description = (
        "Explore a procedural isometric dungeon. Battle enemies, collect gold, and defeat the final boss."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 1000
    FINAL_BOSS_COUNT = 5
    ROOMS_PER_LEVEL = 10

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_FLOOR = (50, 50, 60)
    COLOR_FLOOR_EDGE = (70, 70, 80)
    COLOR_WALL = (40, 40, 50)
    COLOR_WALL_EDGE = (60, 60, 70)
    COLOR_DOOR = (100, 80, 50)
    COLOR_DOOR_EDGE = (120, 100, 70)
    
    COLOR_PLAYER = (60, 150, 255)
    COLOR_PLAYER_EDGE = (120, 200, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_EDGE = (255, 140, 140)
    COLOR_BOSS = (200, 50, 200)
    COLOR_BOSS_EDGE = (240, 100, 240)
    COLOR_GOLD = (255, 223, 0)
    COLOR_GOLD_EDGE = (255, 255, 100)
    
    COLOR_HEALTH_BG = (100, 0, 0)
    COLOR_HEALTH_FG = (0, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SWORD = (200, 200, 200)

    # Isometric Grid
    TILE_WIDTH = 48
    TILE_HEIGHT = 24
    GRID_SIZE_X = 9
    GRID_SIZE_Y = 9

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        self.iso_origin_x = self.SCREEN_WIDTH // 2
        self.iso_origin_y = 100

        self.player = None
        self.enemies = []
        self.golds = []
        self.particles = []

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminated_reason = ""

        self.player_max_health = 20
        self.player = {
            "x": self.GRID_SIZE_X // 2,
            "y": self.GRID_SIZE_Y - 2,
            "health": self.player_max_health,
            "is_defending": False,
            "damage": 2,
        }

        self.current_room = 1
        self.bosses_defeated = 0
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.particles.clear()
        self.player["is_defending"] = False

        # 1. Parse player action (Attack > Move > Defend)
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False
        # --- Player Turn ---
        if space_held: # Attack
            action_taken = True
            # Find adjacent enemy
            target_enemy = None
            for enemy in self.enemies:
                if abs(enemy["x"] - self.player["x"]) + abs(enemy["y"] - self.player["y"]) == 1:
                    target_enemy = enemy
                    break
            
            if target_enemy:
                # Player attacks enemy
                damage = self.player["damage"]
                target_enemy["health"] -= damage
                self.particles.append(self._create_damage_particle(target_enemy, damage))
                # Sfx: sword_hit.wav
                
                if target_enemy["health"] <= 0:
                    reward += 10 if target_enemy["is_boss"] else 5
                    self.score += 10 if target_enemy["is_boss"] else 5
                    self.enemies.remove(target_enemy)
                    # Sfx: enemy_die.wav
            
            # Create sword swing animation regardless of hit
            self.particles.append(self._create_sword_swing_particle())
            # Sfx: sword_swing.wav

        elif movement != 0: # Move
            action_taken = True
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up (iso)
            elif movement == 2: dy = 1 # Down (iso)
            elif movement == 3: dx = -1 # Left (iso)
            elif movement == 4: dx = 1 # Right (iso)

            new_x, new_y = self.player["x"] + dx, self.player["y"] + dy

            # Check boundaries
            if 0 <= new_x < self.GRID_SIZE_X and 0 <= new_y < self.GRID_SIZE_Y:
                # Check for enemy collision
                can_move = not any(e["x"] == new_x and e["y"] == new_y for e in self.enemies)
                
                if can_move:
                    self.player["x"], self.player["y"] = new_x, new_y
                    # Sfx: player_step.wav
                    
                    # Check for gold
                    for gold in self.golds[:]:
                        if gold["x"] == new_x and gold["y"] == new_y:
                            reward += gold["value"]
                            self.score += gold["value"]
                            self.golds.remove(gold)
                            # Sfx: gold_pickup.wav
                    
                    # Check for door transition
                    if self.player["y"] == 0 and self.player["x"] == self.GRID_SIZE_X // 2 and not self.enemies:
                        self._transition_to_next_level()
                        # Sfx: door_open.wav
                        
        else: # Defend (no-op or shift)
            action_taken = True
            self.player["is_defending"] = True
            reward -= 0.2
            # Sfx: defend_shield.wav

        # --- Enemy Turn ---
        if action_taken:
            for enemy in self.enemies:
                enemy["attack_cooldown"] = max(0, enemy["attack_cooldown"] - 1)
                dist_x = self.player["x"] - enemy["x"]
                dist_y = self.player["y"] - enemy["y"]
                
                if abs(dist_x) + abs(dist_y) == 1: # Attack if adjacent
                    damage = enemy["damage"]
                    if enemy["attack_cooldown"] == 0:
                        damage *= 2 # Special attack
                        enemy["attack_cooldown"] = 3
                        self.particles.append(self._create_damage_particle(self.player, damage, is_special=True))
                    else:
                        self.particles.append(self._create_damage_particle(self.player, damage))
                    
                    if self.player["is_defending"]:
                        damage = math.ceil(damage / 2)
                    
                    self.player["health"] -= damage
                    # Sfx: player_hurt.wav
                
                else: # Move towards player
                    move_x, move_y = 0, 0
                    if abs(dist_x) > abs(dist_y):
                        move_x = np.sign(dist_x)
                    else:
                        move_y = np.sign(dist_y)
                    
                    new_ex, new_ey = enemy["x"] + move_x, enemy["y"] + move_y
                    is_occupied = any(e["x"] == new_ex and e["y"] == new_ey for e in self.enemies)
                    is_player = self.player["x"] == new_ex and self.player["y"] == new_ey
                    if not is_occupied and not is_player:
                        enemy["x"], enemy["y"] = new_ex, new_ey

        # --- Termination Check ---
        terminated = False
        if self.player["health"] <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.terminated_reason = "YOU DIED"
            # Sfx: game_over.wav
        elif self.bosses_defeated >= self.FINAL_BOSS_COUNT:
            reward += 100
            terminated = True
            self.game_over = True
            self.terminated_reason = "YOU WIN!"
            # Sfx: victory.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.terminated_reason = "TIME OUT"
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _transition_to_next_level(self):
        if self.current_room % self.ROOMS_PER_LEVEL == 0:
            self.bosses_defeated += 1
        self.current_room += 1
        self.player["x"] = self.GRID_SIZE_X // 2
        self.player["y"] = self.GRID_SIZE_Y - 2
        self._generate_level()

    def _generate_level(self):
        self.enemies.clear()
        self.golds.clear()
        
        occupied_coords = set([(self.player["x"], self.player["y"])])
        
        is_boss_room = self.current_room % self.ROOMS_PER_LEVEL == 0 and self.bosses_defeated < self.FINAL_BOSS_COUNT
        
        if is_boss_room:
            boss_health = 20 * (1.5 ** self.bosses_defeated)
            boss_damage = 3 + self.bosses_defeated
            self.enemies.append({
                "x": self.GRID_SIZE_X // 2, "y": 2,
                "health": boss_health, "max_health": boss_health,
                "damage": boss_damage, "is_boss": True, "attack_cooldown": 3,
            })
            occupied_coords.add((self.GRID_SIZE_X // 2, 2))
        else:
            num_enemies = self.np_random.integers(1, 4)
            for _ in range(num_enemies):
                for _ in range(100): # Max attempts to place
                    x, y = self.np_random.integers(0, self.GRID_SIZE_X), self.np_random.integers(1, self.GRID_SIZE_Y - 2)
                    if (x, y) not in occupied_coords:
                        enemy_health = 5 + (self.current_room - 1)
                        self.enemies.append({
                            "x": x, "y": y,
                            "health": enemy_health, "max_health": enemy_health,
                            "damage": 1 + (self.current_room // 5),
                            "is_boss": False, "attack_cooldown": 3,
                        })
                        occupied_coords.add((x, y))
                        break
        
        num_golds = self.np_random.integers(2, 6)
        for _ in range(num_golds):
            for _ in range(100):
                x, y = self.np_random.integers(0, self.GRID_SIZE_X), self.np_random.integers(0, self.GRID_SIZE_Y)
                if (x, y) not in occupied_coords:
                    gold_value = 1 + (self.current_room // 2)
                    self.golds.append({"x": x, "y": y, "value": gold_value})
                    occupied_coords.add((x, y))
                    break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "current_room": self.current_room,
            "bosses_defeated": self.bosses_defeated,
        }

    # --- Rendering ---

    def _cart_to_iso(self, x, y):
        iso_x = self.iso_origin_x + (x - y) * (self.TILE_WIDTH / 2)
        iso_y = self.iso_origin_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, x, y, color, edge_color, height):
        iso_x, iso_y = self._cart_to_iso(x, y)
        tile_w_half, tile_h_half = self.TILE_WIDTH / 2, self.TILE_HEIGHT / 2
        
        points_top = [
            (iso_x, iso_y - height),
            (iso_x + tile_w_half, iso_y + tile_h_half - height),
            (iso_x, iso_y + self.TILE_HEIGHT - height),
            (iso_x - tile_w_half, iso_y + tile_h_half - height)
        ]
        
        points_left = [
            (iso_x - tile_w_half, iso_y + tile_h_half - height),
            (iso_x, iso_y + self.TILE_HEIGHT - height),
            (iso_x, iso_y + self.TILE_HEIGHT),
            (iso_x - tile_w_half, iso_y + tile_h_half)
        ]

        points_right = [
            (iso_x + tile_w_half, iso_y + tile_h_half - height),
            (iso_x, iso_y + self.TILE_HEIGHT - height),
            (iso_x, iso_y + self.TILE_HEIGHT),
            (iso_x + tile_w_half, iso_y + tile_h_half)
        ]
        
        # Draw polygons
        pygame.gfxdraw.filled_polygon(surface, points_right, [c * 0.7 for c in color])
        pygame.gfxdraw.aapolygon(surface, points_right, [c * 0.7 for c in edge_color])
        pygame.gfxdraw.filled_polygon(surface, points_left, [c * 0.9 for c in color])
        pygame.gfxdraw.aapolygon(surface, points_left, [c * 0.9 for c in edge_color])
        pygame.gfxdraw.filled_polygon(surface, points_top, color)
        pygame.gfxdraw.aapolygon(surface, points_top, edge_color)

    def _draw_health_bar(self, surface, pos, health, max_health):
        health_ratio = max(0, health / max_health)
        bar_width = 30
        bar_height = 5
        x, y = pos[0] - bar_width / 2, pos[1] - 35
        
        pygame.draw.rect(surface, self.COLOR_HEALTH_BG, (x, y, bar_width, bar_height))
        pygame.draw.rect(surface, self.COLOR_HEALTH_FG, (x, y, int(bar_width * health_ratio), bar_height))

    def _render_game(self):
        # Draw floor and walls
        for y in range(self.GRID_SIZE_Y):
            for x in range(self.GRID_SIZE_X):
                is_door = (x == self.GRID_SIZE_X // 2 and y == 0) and not self.enemies
                color = self.COLOR_DOOR if is_door else self.COLOR_FLOOR
                edge_color = self.COLOR_DOOR_EDGE if is_door else self.COLOR_FLOOR_EDGE
                self._draw_iso_cube(self.screen, x, y, color, edge_color, 0)
        
        # Draw entities in correct Z-order
        entities = sorted(
            [("gold", g) for g in self.golds] +
            [("enemy", e) for e in self.enemies] +
            [("player", self.player)],
            key=lambda item: item[1]['x'] + item[1]['y']
        )
        
        for type, entity in entities:
            height, color, edge_color = 0, (0,0,0), (0,0,0)
            if type == "gold":
                height, color, edge_color = 5, self.COLOR_GOLD, self.COLOR_GOLD_EDGE
            elif type == "player":
                height, color, edge_color = 18, self.COLOR_PLAYER, self.COLOR_PLAYER_EDGE
            elif type == "enemy":
                if entity["is_boss"]:
                    height, color, edge_color = 24, self.COLOR_BOSS, self.COLOR_BOSS_EDGE
                else:
                    height, color, edge_color = 16, self.COLOR_ENEMY, self.COLOR_ENEMY_EDGE

            self._draw_iso_cube(self.screen, entity["x"], entity["y"], color, edge_color, height)
            
            if type in ["player", "enemy"]:
                iso_x, iso_y = self._cart_to_iso(entity["x"], entity["y"])
                max_h = self.player_max_health if type == "player" else entity["max_health"]
                self._draw_health_bar(self.screen, (iso_x, iso_y - height), entity["health"], max_h)

        # Draw particles
        for p in self.particles:
            if p["type"] == "damage":
                text = self.font_small.render(str(p["value"]), True, p["color"])
                self.screen.blit(text, (p["x"], p["y"]))
            elif p["type"] == "sword":
                pygame.draw.aaline(self.screen, self.COLOR_SWORD, p["start"], p["end"], 2)


    def _render_ui(self):
        # Player Health
        health_text = self.font_medium.render(f"HP: {max(0, self.player['health'])} / {self.player_max_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Score / Gold
        score_text = self.font_medium.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        self.screen.blit(score_text, (10, 40))

        # Room / Bosses
        level_text = self.font_medium.render(f"Room: {self.current_room}", True, self.COLOR_TEXT)
        boss_text = self.font_medium.render(f"Bosses: {self.bosses_defeated}/{self.FINAL_BOSS_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - 150, 10))
        self.screen.blit(boss_text, (self.SCREEN_WIDTH - 150, 40))
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.terminated_reason, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    # --- Particle Helpers ---
    def _create_damage_particle(self, target, value, is_special=False):
        iso_x, iso_y = self._cart_to_iso(target["x"], target["y"])
        color = (255, 100, 0) if is_special else (255, 255, 255)
        return {"type": "damage", "x": iso_x - 5, "y": iso_y - 40, "value": value, "color": color}

    def _create_sword_swing_particle(self):
        px, py = self.player["x"], self.player["y"]
        iso_x, iso_y = self._cart_to_iso(px, py)
        return {
            "type": "sword",
            "start": (iso_x - 15, iso_y - 25),
            "end": (iso_x + 15, iso_y - 15),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert self.player['health'] == self.player_max_health
        assert self.score == 0
        assert self.current_room == 1
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use arrow keys for movement, space for attack, shift to defend
    pygame.display.set_caption("Isometric Dungeon Crawler")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop for manual play
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        # Since it's turn-based, we only step on a key press event
        action_taken = any(k for k in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT] if keys[k])
        
        # To make it playable, we need a small delay and only process one action per "turn"
        # The environment itself is stateless between steps, but for a human player, we need to register a single action and then step.
        # This part is tricky. A better way for human play is to wait for a keydown event.
        
        # Simplified manual play loop: step on any key press
        wait_for_action = True
        while wait_for_action:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    wait_for_action = False
                if event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]: movement = 1
                    elif keys[pygame.K_DOWN]: movement = 2
                    elif keys[pygame.K_LEFT]: movement = 3
                    elif keys[pygame.K_RIGHT]: movement = 4

                    if keys[pygame.K_SPACE]: space = 1
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                    
                    wait_for_action = False
            
            # Draw the current state while waiting
            obs_for_display = np.transpose(env._get_observation(), (1, 0, 2))
            surf = pygame.surfarray.make_surface(obs_for_display)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated: break
        
        if terminated: break

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")

    print("Game Over!")
    env.close()