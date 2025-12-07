
# Generated: 2025-08-28T00:27:47.153369
# Source Brief: brief_03792.md
# Brief Index: 3792

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move, Space to attack, Shift to use a health potion. "
        "Actions are turn-based."
    )

    game_description = (
        "Explore an isometric dungeon, fight monsters, and defeat the final boss. "
        "Each action is a turn. Good luck!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self._init_constants()
        self._init_fonts()

        self.reset()
        self.validate_implementation()

    def _init_constants(self):
        self.COLOR_BG = (18, 22, 33)
        self.COLOR_WALL_TOP = (60, 68, 87)
        self.COLOR_WALL_SIDE = (49, 56, 70)
        self.COLOR_FLOOR = (92, 78, 65)
        self.COLOR_DOOR = (143, 110, 76)
        
        self.COLOR_PLAYER = (52, 217, 134)
        self.COLOR_PLAYER_OUTLINE = (220, 255, 230)
        self.COLOR_ENEMY = (217, 52, 82)
        self.COLOR_ENEMY_OUTLINE = (255, 220, 230)
        self.COLOR_BOSS = (168, 52, 217)
        self.COLOR_BOSS_OUTLINE = (240, 220, 255)
        self.COLOR_POTION = (52, 168, 217)
        
        self.COLOR_HEALTH_BG = (80, 20, 20)
        self.COLOR_HEALTH_FG = (20, 200, 20)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_POPUP = (255, 200, 50)
        self.COLOR_SHADOW = (0, 0, 0, 90)

        self.ROOM_WIDTH, self.ROOM_HEIGHT = 10, 12
        self.NUM_ROOMS = 5
        self.MAX_STEPS = 1000
        
        self.REWARD_HIT_ENEMY = 0.1
        self.REWARD_DAMAGE_TAKEN = 0.1
        self.REWARD_POTION = 1.0
        self.REWARD_KILL_ENEMY = 5.0
        self.REWARD_NEW_ROOM = 10.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0

        self.TILE_W, self.TILE_H = 48, 24
        self.TILE_W_HALF, self.TILE_H_HALF = self.TILE_W // 2, self.TILE_H // 2
        self.ISO_Z = 18 # Height of wall cubes

    def _init_fonts(self):
        self.font_ui = pygame.font.Font(None, 24)
        self.font_popup = pygame.font.Font(None, 20)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.current_room_idx = 0
        self.damage_popups = []

        self._generate_dungeon()
        
        self.player = {
            "x": self.ROOM_WIDTH // 2, "y": self.ROOM_HEIGHT - 2,
            "hp": 20, "max_hp": 20, "dmg": 3,
            "potions": 1, "xp": 0,
            "facing": (0, -1), # Up
            "type": "player"
        }
        
        self.current_room = self.rooms[self.current_room_idx]

        return self._get_observation(), self._get_info()

    def _generate_dungeon(self):
        self.rooms = []
        for i in range(self.NUM_ROOMS):
            is_boss_room = (i == self.NUM_ROOMS - 1)
            entities = []
            
            valid_spawns = []
            for y in range(2, self.ROOM_HEIGHT - 2):
                for x in range(1, self.ROOM_WIDTH - 1):
                    valid_spawns.append((x, y))
            self.np_random.shuffle(valid_spawns)

            if is_boss_room:
                pos = (self.ROOM_WIDTH // 2, self.ROOM_HEIGHT // 3)
                entities.append({
                    "x": pos[0], "y": pos[1], "hp": 50, "max_hp": 50,
                    "dmg": 5, "type": "boss", "patrol_dir": (1,0)
                })
            else:
                num_enemies = self.np_random.integers(i, i + 2)
                for _ in range(num_enemies):
                    if not valid_spawns: break
                    pos = valid_spawns.pop()
                    entities.append({
                        "x": pos[0], "y": pos[1], "hp": 5 + i * 2, "max_hp": 5 + i * 2,
                        "dmg": 1 + i, "type": "enemy",
                        "patrol_dir": self.np_random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                    })
                
                if not is_boss_room and self.np_random.random() < 0.7:
                    if not valid_spawns: break
                    pos = valid_spawns.pop()
                    entities.append({"x": pos[0], "y": pos[1], "type": "potion", "value": 10})
            
            self.rooms.append({"entities": entities, "door_pos": (self.ROOM_WIDTH // 2, 0)})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.damage_popups.clear()
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # 1. Player Action
        action_taken = True
        if shift_pressed:
            if self.player["potions"] > 0 and self.player["hp"] < self.player["max_hp"]:
                self.player["potions"] -= 1
                heal_amount = self.np_random.integers(8, 13)
                self.player["hp"] = min(self.player["max_hp"], self.player["hp"] + heal_amount)
                reward += self.REWARD_POTION
                self._add_popup(f"+{heal_amount} HP", self.player)
                # sfx: potion drink
        elif space_pressed:
            target_x = self.player["x"] + self.player["facing"][0]
            target_y = self.player["y"] + self.player["facing"][1]
            
            target_enemy = None
            for enemy in self.current_room["entities"]:
                if enemy.get("hp", 0) > 0 and enemy["x"] == target_x and enemy["y"] == target_y:
                    target_enemy = enemy
                    break
            
            if target_enemy:
                damage = self.player["dmg"]
                target_enemy["hp"] -= damage
                reward += self.REWARD_HIT_ENEMY
                self._add_popup(f"-{damage}", target_enemy)
                # sfx: sword swing and hit
                if target_enemy["hp"] <= 0:
                    reward += self.REWARD_KILL_ENEMY
                    self.player["xp"] += 10
                    if target_enemy["type"] == "boss":
                        self.game_over = True
                        self.termination_reason = "Victory!"
                    # sfx: enemy defeat
        elif movement > 0:
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_map[movement]
            self.player["facing"] = (dx, dy)
            
            nx, ny = self.player["x"] + dx, self.player["y"] + dy

            # Collision detection
            is_wall = not (0 <= nx < self.ROOM_WIDTH and 0 <= ny < self.ROOM_HEIGHT)
            is_occupied = any(e["x"] == nx and e["y"] == ny and e.get("hp", 0) > 0 for e in self.current_room["entities"])
            
            if not is_wall and not is_occupied:
                self.player["x"], self.player["y"] = nx, ny
                # sfx: footstep
        else:
            action_taken = False # No-op

        # 2. Post-move checks (potions, doors)
        # Check for potion collection
        collected_potion = None
        for entity in self.current_room["entities"]:
            if entity["type"] == "potion" and entity["x"] == self.player["x"] and entity["y"] == self.player["y"]:
                collected_potion = entity
                break
        if collected_potion:
            self.player["potions"] += 1
            reward += self.REWARD_POTION
            self._add_popup("+1 Potion", self.player)
            self.current_room["entities"].remove(collected_potion)
            # sfx: item pickup

        # Check for door transition
        if (self.player["x"], self.player["y"]) == self.current_room["door_pos"]:
            if self.current_room_idx < self.NUM_ROOMS - 1:
                self.current_room_idx += 1
                self.current_room = self.rooms[self.current_room_idx]
                self.player["y"] = self.ROOM_HEIGHT - 2 # Enter from bottom
                reward += self.REWARD_NEW_ROOM
                # sfx: door open
        
        # 3. Enemy Turn
        if action_taken:
            for enemy in self.current_room["entities"]:
                if enemy.get("hp", 0) <= 0: continue

                is_adjacent = abs(enemy["x"] - self.player["x"]) + abs(enemy["y"] - self.player["y"]) == 1
                
                if is_adjacent:
                    num_attacks = 2 if enemy["type"] == "boss" else 1
                    for _ in range(num_attacks):
                        if self.player["hp"] > 0:
                            damage = enemy["dmg"]
                            self.player["hp"] -= damage
                            reward -= damage * self.REWARD_DAMAGE_TAKEN
                            self._add_popup(f"-{damage}", self.player)
                            # sfx: player hurt
                else: # Move towards player
                    p_dx, p_dy = self.player["x"] - enemy["x"], self.player["y"] - enemy["y"]
                    if abs(p_dx) > abs(p_dy):
                        ex, ey = enemy["x"] + np.sign(p_dx), enemy["y"]
                    else:
                        ex, ey = enemy["x"], enemy["y"] + np.sign(p_dy)
                    
                    is_wall = not (0 <= ex < self.ROOM_WIDTH and 0 <= ey < self.ROOM_HEIGHT)
                    is_occupied = any(e["x"] == ex and e["y"] == ey and e.get("hp", 0) > 0 for e in self.current_room["entities"])
                    if not is_wall and not is_occupied and (ex, ey) != (self.player['x'], self.player['y']):
                        enemy["x"], enemy["y"] = ex, ey

        # Clean up defeated enemies
        self.current_room["entities"] = [e for e in self.current_room["entities"] if e.get("hp", -1) > 0 or e["type"] == "potion"]

        # 4. Update Game State
        self.steps += 1
        self.score += reward

        if self.player["hp"] <= 0:
            self.game_over = True
            self.termination_reason = "You Died"
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.termination_reason = "Time Limit Reached"

        if terminated:
            if self.termination_reason == "Victory!":
                reward += self.REWARD_WIN
            elif self.termination_reason == "You Died":
                reward += self.REWARD_LOSE
            self.score += reward # Add terminal reward to final score

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _add_popup(self, text, target):
        self.damage_popups.append({"text": text, "x": target["x"], "y": target["y"], "timer": 1})

    def _to_iso(self, x, y):
        origin_x = self.width // 2
        origin_y = self.height // 2 - self.ROOM_HEIGHT * self.TILE_H_HALF // 2
        screen_x = origin_x + (x - y) * self.TILE_W_HALF
        screen_y = origin_y + (x + y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, screen_x, screen_y, top_color, side_color, z_height):
        points_top = [
            (screen_x, screen_y - z_height),
            (screen_x + self.TILE_W_HALF, screen_y + self.TILE_H_HALF - z_height),
            (screen_x, screen_y + self.TILE_H - z_height),
            (screen_x - self.TILE_W_HALF, screen_y + self.TILE_H_HALF - z_height),
        ]
        points_left = [
            (screen_x - self.TILE_W_HALF, screen_y + self.TILE_H_HALF - z_height),
            (screen_x, screen_y + self.TILE_H - z_height),
            (screen_x, screen_y + self.TILE_H),
            (screen_x - self.TILE_W_HALF, screen_y + self.TILE_H_HALF),
        ]
        points_right = [
            (screen_x + self.TILE_W_HALF, screen_y + self.TILE_H_HALF - z_height),
            (screen_x, screen_y + self.TILE_H - z_height),
            (screen_x, screen_y + self.TILE_H),
            (screen_x + self.TILE_W_HALF, screen_y + self.TILE_H_HALF),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points_top, top_color)
        pygame.gfxdraw.aapolygon(self.screen, points_top, top_color)
        pygame.gfxdraw.filled_polygon(self.screen, points_left, side_color)
        pygame.gfxdraw.aapolygon(self.screen, points_left, side_color)
        pygame.gfxdraw.filled_polygon(self.screen, points_right, side_color)
        pygame.gfxdraw.aapolygon(self.screen, points_right, side_color)

    def _draw_iso_tile(self, screen_x, screen_y, color):
        points = [
            (screen_x, screen_y),
            (screen_x + self.TILE_W_HALF, screen_y + self.TILE_H_HALF),
            (screen_x, screen_y + self.TILE_H),
            (screen_x - self.TILE_W_HALF, screen_y + self.TILE_H_HALF),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_game(self):
        # Draw floor and walls
        door_pos = self.current_room["door_pos"]
        for y in range(self.ROOM_HEIGHT):
            for x in range(self.ROOM_WIDTH):
                sx, sy = self._to_iso(x, y)
                color = self.COLOR_DOOR if (x,y) == door_pos else self.COLOR_FLOOR
                self._draw_iso_tile(sx, sy, color)
        for y in range(self.ROOM_HEIGHT):
            for x in [-1, self.ROOM_WIDTH]:
                sx, sy = self._to_iso(x, y)
                self._draw_iso_cube(sx, sy, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE, self.ISO_Z)
        for x in range(self.ROOM_WIDTH):
             for y in [-1, self.ROOM_HEIGHT]:
                sx, sy = self._to_iso(x, y)
                self._draw_iso_cube(sx, sy, self.COLOR_WALL_TOP, self.COLOR_WALL_SIDE, self.ISO_Z)

        # Draw entities
        all_entities = [self.player] + self.current_room["entities"]
        sorted_entities = sorted(all_entities, key=lambda e: (e["y"], e["x"]))

        for entity in sorted_entities:
            sx, sy = self._to_iso(entity["x"], entity["y"])
            bob = int(math.sin(pygame.time.get_ticks() * 0.005 + entity["x"]) * 3)
            
            # Shadow
            shadow_surface = pygame.Surface((self.TILE_W, self.TILE_H), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, [0, 0, self.TILE_W, self.TILE_H])
            self.screen.blit(shadow_surface, (sx - self.TILE_W_HALF, sy))

            if entity["type"] == "player":
                self._draw_sprite(sx, sy - bob, self.COLOR_PLAYER, self.COLOR_PLAYER_OUTLINE)
            elif entity["type"] == "enemy":
                self._draw_sprite(sx, sy - bob, self.COLOR_ENEMY, self.COLOR_ENEMY_OUTLINE)
            elif entity["type"] == "boss":
                self._draw_sprite(sx, sy - bob, self.COLOR_BOSS, self.COLOR_BOSS_OUTLINE, is_boss=True)
            elif entity["type"] == "potion":
                self._draw_potion(sx, sy - bob)

            # Health bar
            if "hp" in entity and entity["hp"] < entity["max_hp"]:
                hp_ratio = max(0, entity["hp"] / entity["max_hp"])
                bar_y = sy - self.ISO_Z - 15 - bob
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (sx - 15, bar_y, 30, 5))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (sx - 15, bar_y, 30 * hp_ratio, 5))
        
        # Draw popups
        for popup in self.damage_popups:
            sx, sy = self._to_iso(popup["x"], popup["y"])
            text_surf = self.font_popup.render(popup["text"], True, self.COLOR_POPUP)
            text_rect = text_surf.get_rect(center=(sx, sy - 30 - self.steps % 10))
            self.screen.blit(text_surf, text_rect)

    def _draw_sprite(self, sx, sy, color, outline_color, is_boss=False):
        z_offset = self.ISO_Z
        if is_boss:
            points = [
                (sx, sy - z_offset - 5), (sx + 12, sy - z_offset + 10),
                (sx, sy - z_offset + 25), (sx - 12, sy - z_offset + 10)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)
            # Horns
            pygame.draw.line(self.screen, outline_color, (sx - 3, sy - z_offset - 3), (sx - 8, sy - z_offset - 12), 2)
            pygame.draw.line(self.screen, outline_color, (sx + 3, sy - z_offset - 3), (sx + 8, sy - z_offset - 12), 2)
        else:
            points = [
                (sx, sy - z_offset), (sx + 10, sy - z_offset + 8),
                (sx, sy - z_offset + 16), (sx - 10, sy - z_offset + 8)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _draw_potion(self, sx, sy):
        z_offset = self.ISO_Z - 5
        pygame.draw.rect(self.screen, self.COLOR_POTION, (sx - 4, sy - z_offset, 8, 10))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (sx - 2, sy - z_offset - 3, 4, 3))

    def _render_ui(self):
        # Info Panel
        pygame.draw.rect(self.screen, (0,0,0,150), (5, 5, 200, 100))
        
        hp_text = f"HP: {self.player['hp']} / {self.player['max_hp']}"
        pot_text = f"Potions: {self.player['potions']}"
        xp_text = f"XP: {self.player['xp']}"
        room_text = f"Dungeon Level: {self.current_room_idx + 1} / {self.NUM_ROOMS}"

        self._draw_text(hp_text, (15, 15))
        self._draw_text(pot_text, (15, 35))
        self._draw_text(xp_text, (15, 55))
        self._draw_text(room_text, (15, 75))

        if self.game_over:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            end_font = pygame.font.Font(None, 60)
            text_surf = end_font.render(self.termination_reason, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, color=None):
        if color is None: color = self.COLOR_TEXT
        text_surface = self.font_ui.render(text, True, color)
        self.screen.blit(text_surface, pos)

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
            "player_hp": self.player["hp"],
            "potions": self.player["potions"],
            "xp": self.player["xp"],
            "room": self.current_room_idx,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Isometric Dungeon Crawler")
    screen = pygame.display.set_mode((env.width, env.height))
    clock = pygame.time.Clock()
    
    print(env.user_guide)

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    action.fill(0)

        if not done:
            keys = pygame.key.get_pressed()
            
            # Reset action
            action.fill(0)

            # This logic prioritizes one action per frame for turn-based play
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
            elif keys[pygame.K_SPACE]:
                action[1] = 1
            elif keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Only step if an action is taken
            if action.any():
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

    env.close()