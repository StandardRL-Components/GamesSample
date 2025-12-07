
# Generated: 2025-08-27T18:27:40.143803
# Source Brief: brief_01836.md
# Brief Index: 1836

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. In combat, Space to attack, Shift to use a special attack (costs XP). "
        "No-op (no keys) to defend."
    )

    game_description = (
        "Explore a 5-room dungeon, defeat enemies in turn-based combat, and slay the final boss. "
        "Collect gold and XP to grow stronger."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.TILE_SIZE = self.WIDTH // self.GRID_WIDTH
        self.MAX_STEPS = 2000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (70, 70, 80)
        self.COLOR_FLOOR = (40, 40, 50)
        self.COLOR_DOOR_CLOSED = (100, 60, 40)
        self.COLOR_DOOR_OPEN = (140, 90, 60)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_PLAYER_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_ENEMY_DMG = (255, 200, 200)
        self.COLOR_BOSS = (150, 0, 0)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_XP = (0, 150, 255)
        self.COLOR_POTION = (0, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_FRAME = (90, 90, 110)
        self.COLOR_LOG_TEXT = (200, 200, 200)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # --- State variables will be initialized in reset() ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.player = {}
        self.dungeon = []
        self.current_room_idx = 0
        self.game_state = "EXPLORE" # or "COMBAT"
        self.combat_enemy = None
        self.message_log = deque(maxlen=4)
        self.particles = []
        self.damage_flash_timer = 0
        self.damaged_entity = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_state = "EXPLORE"
        self.combat_enemy = None
        self.message_log.clear()
        self.particles = []
        self.damage_flash_timer = 0
        self.damaged_entity = None

        self.player = {
            "pos": [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2],
            "hp": 100,
            "max_hp": 100,
            "xp": 0,
            "gold": 0,
            "attack": 10,
            "is_defending": False,
        }

        self._generate_dungeon()
        self.current_room_idx = 0
        self.player['pos'] = [3, self.GRID_HEIGHT // 2]
        self.message_log.append("You enter the dungeon. Find the boss!")

        return self._get_observation(), self._get_info()

    def _generate_dungeon(self):
        self.dungeon = []
        for i in range(5):
            room = {
                "grid": np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int),
                "enemies": [],
                "items": [],
                "is_cleared": False,
                "is_boss_room": (i == 4)
            }
            # Walls
            room["grid"][0, :] = 1
            room["grid"][-1, :] = 1
            room["grid"][:, 0] = 1
            room["grid"][:, -1] = 1

            # Doors (linear progression)
            if i > 0: # Entrance
                room["grid"][0, self.GRID_HEIGHT // 2 -1 : self.GRID_HEIGHT // 2 + 2] = 2
            if i < 4: # Exit
                room["grid"][-1, self.GRID_HEIGHT // 2 -1 : self.GRID_HEIGHT // 2 + 2] = 2

            # Populate
            if not room["is_boss_room"]:
                self._populate_room(room, i)
            else:
                self._populate_boss_room(room, i)

            self.dungeon.append(room)

    def _populate_room(self, room, room_idx):
        num_enemies = self.np_random.integers(2, 5)
        num_items = self.np_random.integers(3, 6)
        difficulty_mod = 1.0 + (room_idx * 0.1)

        for _ in range(num_enemies):
            pos = self._get_random_empty_pos(room)
            if pos:
                enemy_type = self.np_random.choice(['goblin', 'skeleton', 'slime'])
                if enemy_type == 'goblin':
                    hp, attack = 20, 5
                elif enemy_type == 'skeleton':
                    hp, attack = 30, 3
                else: # slime
                    hp, attack = 15, 7

                room["enemies"].append({
                    "pos": pos,
                    "hp": int(hp * difficulty_mod), "max_hp": int(hp * difficulty_mod),
                    "attack": int(attack * difficulty_mod), "type": enemy_type,
                    "xp_reward": int(10 * difficulty_mod), "gold_reward": self.np_random.integers(3, 8),
                    "attack_cycle": 0
                })

        for _ in range(num_items):
            pos = self._get_random_empty_pos(room)
            if pos:
                item_type = self.np_random.choice(['gold', 'xp', 'potion'])
                room["items"].append({"pos": pos, "type": item_type})

    def _populate_boss_room(self, room, room_idx):
        difficulty_mod = 1.0 + (room_idx * 0.1)
        pos = [self.GRID_WIDTH - 5, self.GRID_HEIGHT // 2]
        room["enemies"].append({
            "pos": pos, "type": "boss",
            "hp": int(150 * difficulty_mod), "max_hp": int(150 * difficulty_mod),
            "attack": int(15 * difficulty_mod),
            "xp_reward": 100, "gold_reward": 50,
            "attack_cycle": 0
        })

    def _get_random_empty_pos(self, room):
        attempts = 50
        for _ in range(attempts):
            x = self.np_random.integers(5, self.GRID_WIDTH - 5)
            y = self.np_random.integers(2, self.GRID_HEIGHT - 2)
            is_occupied = any(e['pos'] == [x, y] for e in room['enemies']) or \
                          any(i['pos'] == [x, y] for i in room['items'])
            if not is_occupied:
                return [x, y]
        return None

    def step(self, action):
        reward = 0
        terminated = False
        self.game_over = False
        self.player['is_defending'] = False
        self.message_log.clear()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.game_state == "EXPLORE":
            if movement != 0:
                reward += self._handle_movement(movement)
            else: # No-op in explore does nothing
                pass

        elif self.game_state == "COMBAT":
            reward += self._handle_combat(movement, space_held, shift_held)

        # Update game systems
        self._update_particles()
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1
            if self.damage_flash_timer == 0:
                self.damaged_entity = None

        # Check for termination conditions
        if self.player["hp"] <= 0:
            self.message_log.append("You have been defeated.")
            reward -= 100
            terminated = True
            self.game_over = True
        
        # Check for boss defeat
        if self.dungeon[4]['is_cleared']:
            self.message_log.append("You defeated the final boss! Victory!")
            reward += 100
            terminated = True
            self.game_over = True

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over:
                self.message_log.append("Time is up!")

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        px, py = self.player["pos"]
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
        nx, ny = px + dx, py + dy

        current_room = self.dungeon[self.current_room_idx]
        tile = current_room["grid"][nx, ny]

        # Wall collision
        if tile == 1:
            return 0

        # Door collision
        if tile == 2:
            if current_room["is_cleared"]:
                if nx == 0: # Moving left to previous room
                    self.current_room_idx -= 1
                    self.player["pos"] = [self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2]
                else: # Moving right to next room
                    self.current_room_idx += 1
                    self.player["pos"] = [1, self.GRID_HEIGHT // 2]
                self.message_log.append(f"You enter room {self.current_room_idx + 1}.")
                return 5 # Room clear reward
            else:
                self.message_log.append("Defeat all enemies to proceed.")
                return 0

        # Enemy collision (initiates combat)
        for enemy in current_room["enemies"]:
            if enemy["pos"] == [nx, ny]:
                self.game_state = "COMBAT"
                self.combat_enemy = enemy
                self.message_log.append(f"A wild {enemy['type']} appears!")
                return 0

        # Item pickup
        reward = 0
        for i, item in enumerate(current_room["items"]):
            if item["pos"] == [nx, ny]:
                if item["type"] == "gold":
                    self.player["gold"] += 1
                    reward += 0.1
                    self.message_log.append("+1 Gold!")
                elif item["type"] == "xp":
                    self.player["xp"] += 1
                    reward += 0.2
                    self.message_log.append("+1 XP!")
                elif item["type"] == "potion":
                    heal_amount = 25
                    self.player["hp"] = min(self.player["max_hp"], self.player["hp"] + heal_amount)
                    self.message_log.append(f"Used potion. Healed {heal_amount} HP.")
                
                self._create_particle(f"+1 {item['type'].upper()}", item['pos'], self.COLOR_UI_TEXT)
                current_room["items"].pop(i)
                break
        
        # Move to empty tile
        self.player["pos"] = [nx, ny]
        return reward

    def _handle_combat(self, movement, space_held, shift_held):
        reward = 0
        player_acted = False

        # Player Action
        if shift_held: # Special Attack
            cost = 10
            if self.player['xp'] >= cost:
                self.player['xp'] -= cost
                damage = self.player['attack'] * 2
                self._apply_damage(self.combat_enemy, damage, "player")
                self.message_log.append(f"Special attack hits for {damage} damage!")
                # sfx: special_attack_sound
                player_acted = True
            else:
                self.message_log.append("Not enough XP for special attack!")
                reward -= 0.02 # Penalty for failed action
        elif space_held: # Normal Attack
            damage = self.player['attack'] + self.np_random.integers(-2, 3)
            self._apply_damage(self.combat_enemy, damage, "player")
            self.message_log.append(f"Player attacks for {damage} damage!")
            # sfx: player_attack_sound
            player_acted = True
        elif movement == 0: # Defend
            self.player['is_defending'] = True
            self.message_log.append("Player is defending.")
            reward -= 0.02
            player_acted = True
        else: # Attempted to move
            self.message_log.append("Cannot move during combat!")
            reward -= 0.02
            player_acted = True # Consumes turn

        if not player_acted:
            return 0 # No action taken, wait for next step

        # Check if enemy is defeated
        if self.combat_enemy and self.combat_enemy['hp'] <= 0:
            reward += 1 # Defeat enemy reward
            self.message_log.append(f"The {self.combat_enemy['type']} is defeated!")
            self.player['gold'] += self.combat_enemy['gold_reward']
            self.player['xp'] += self.combat_enemy['xp_reward']
            self.message_log.append(f"Gained {self.combat_enemy['xp_reward']} XP and {self.combat_enemy['gold_reward']} Gold.")

            current_room = self.dungeon[self.current_room_idx]
            current_room['enemies'].remove(self.combat_enemy)
            self.combat_enemy = None
            self.game_state = "EXPLORE"

            if not current_room['enemies']:
                current_room['is_cleared'] = True
                self.message_log.append("Room cleared!")
                if current_room['is_boss_room']:
                    # This state is checked and handled in the main step loop
                    pass
            return reward

        # Enemy Turn (if still in combat)
        if self.game_state == "COMBAT" and player_acted:
            self._enemy_turn()

        return reward

    def _enemy_turn(self):
        enemy = self.combat_enemy
        # Simple cycling attack patterns
        if enemy['type'] == 'goblin': # Quick, normal
            damage = enemy['attack']
        elif enemy['type'] == 'skeleton': # Alternates weak/strong
            damage = enemy['attack'] - 2 if enemy['attack_cycle'] % 2 == 0 else enemy['attack'] + 2
        elif enemy['type'] == 'slime': # Charges up
            damage = enemy['attack'] // 2 if enemy['attack_cycle'] % 3 != 2 else enemy['attack'] * 2
        elif enemy['type'] == 'boss': # More complex pattern
            if enemy['attack_cycle'] % 4 == 3: # Heavy hit
                damage = int(enemy['attack'] * 1.5)
            else: # Normal hit
                damage = enemy['attack']
        else:
            damage = enemy['attack']

        damage += self.np_random.integers(-1, 2)
        
        if self.player['is_defending']:
            damage = int(damage * 0.8) # 20% damage reduction
            self.message_log.append("Defense reduces damage!")

        damage = max(0, damage)
        self._apply_damage(self.player, damage, "enemy")
        self.message_log.append(f"{enemy['type']} attacks for {damage} damage!")
        # sfx: enemy_attack_sound
        enemy['attack_cycle'] += 1

    def _apply_damage(self, target, damage, source):
        damage = max(0, damage)
        target['hp'] -= damage
        target['hp'] = max(0, target['hp'])
        
        self.damage_flash_timer = 5 # frames
        self.damaged_entity = target
        
        pos = target['pos']
        self._create_particle(str(damage), pos, self.COLOR_ENEMY if source == "enemy" else self.COLOR_PLAYER_DMG)

    def _create_particle(self, text, grid_pos, color):
        screen_pos = [(grid_pos[0] + 0.5) * self.TILE_SIZE, (grid_pos[1] + 0.5) * self.TILE_SIZE]
        self.particles.append({
            "text": text,
            "pos": screen_pos,
            "vel": [self.np_random.uniform(-0.5, 0.5), -1.5],
            "timer": 30, # frames
            "color": color
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['timer'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['timer'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        room = self.dungeon[self.current_room_idx]
        ts = self.TILE_SIZE

        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * ts, y * ts, ts, ts)
                tile_type = room["grid"][x, y]
                if tile_type == 1: # Wall
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif tile_type == 2: # Door
                    color = self.COLOR_DOOR_OPEN if room['is_cleared'] else self.COLOR_DOOR_CLOSED
                    pygame.draw.rect(self.screen, color, rect)
                else: # Floor
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)
        
        # Draw items
        for item in room["items"]:
            x, y = item['pos']
            color = self.COLOR_GOLD if item['type'] == 'gold' else \
                    self.COLOR_XP if item['type'] == 'xp' else self.COLOR_POTION
            pygame.draw.circle(self.screen, color, (int((x + 0.5) * ts), int((y + 0.5) * ts)), ts // 3)

        # Draw enemies
        for enemy in room["enemies"]:
            x, y = enemy['pos']
            rect = pygame.Rect(x * ts, y * ts, ts, ts)
            
            color = self.COLOR_BOSS if enemy['type'] == 'boss' else self.COLOR_ENEMY
            if self.damaged_entity is enemy and self.damage_flash_timer > 0:
                color = self.COLOR_ENEMY_DMG

            size_mod = 1.4 if enemy['type'] == 'boss' else 1.0
            draw_rect = rect.inflate(ts * (size_mod - 1), ts * (size_mod - 1))
            pygame.draw.rect(self.screen, color, draw_rect)
            self._render_health_bar(enemy, draw_rect)

        # Draw player
        px, py = self.player['pos']
        player_rect = pygame.Rect(px * ts, py * ts, ts, ts)
        color = self.COLOR_PLAYER
        if self.damaged_entity is self.player and self.damage_flash_timer > 0:
            color = self.COLOR_PLAYER_DMG
        pygame.draw.rect(self.screen, color, player_rect)
        self._render_health_bar(self.player, player_rect)
        
        # Draw combat indicator
        if self.game_state == "COMBAT" and self.combat_enemy:
            ex, ey = self.combat_enemy['pos']
            start_pos = ((px + 0.5) * ts, (py + 0.5) * ts)
            end_pos = ((ex + 0.5) * ts, (ey + 0.5) * ts)
            pygame.draw.line(self.screen, self.COLOR_ENEMY, start_pos, end_pos, 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['timer'] / 30))
            text_surf = self.font_small.render(p['text'], True, p['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(p['pos'][0]), int(p['pos'][1])))

    def _render_health_bar(self, entity, rect):
        if entity['hp'] < entity['max_hp']:
            hp_ratio = entity['hp'] / entity['max_hp']
            bar_width = rect.width
            bar_height = 5
            bar_y = rect.top - bar_height - 2
            
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (rect.left, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (rect.left, bar_y, int(bar_width * hp_ratio), bar_height))

    def _render_ui(self):
        # Player Stats
        stats_text = (f"HP: {self.player['hp']}/{self.player['max_hp']} | "
                      f"XP: {self.player['xp']} | Gold: {self.player['gold']} | "
                      f"Score: {self.score}")
        text_surf = self.font_medium.render(stats_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Boss Health Bar
        if self.dungeon[self.current_room_idx]['is_boss_room']:
            boss = next((e for e in self.dungeon[self.current_room_idx]['enemies'] if e['type'] == 'boss'), None)
            if boss:
                bar_width = self.WIDTH // 2
                bar_height = 15
                x = self.WIDTH // 4
                y = 10
                hp_ratio = boss['hp'] / boss['max_hp']
                
                pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, (x-2, y-2, bar_width+4, bar_height+4))
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (x, y, bar_width, bar_height))
                pygame.draw.rect(self.screen, self.COLOR_BOSS, (x, y, int(bar_width * hp_ratio), bar_height))
                boss_text = self.font_small.render("FINAL BOSS", True, self.COLOR_UI_TEXT)
                self.screen.blit(boss_text, (x + bar_width // 2 - boss_text.get_width() // 2, y))

        # Message Log
        log_y = self.HEIGHT - 20
        for msg in reversed(self.message_log):
            log_surf = self.font_small.render(msg, True, self.COLOR_LOG_TEXT)
            self.screen.blit(log_surf, (10, log_y - log_surf.get_height()))
            log_y -= log_surf.get_height()

        # Game Over/Win Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.player['hp'] > 0 else "GAME OVER"
            text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hp": self.player["hp"],
            "xp": self.player["xp"],
            "gold": self.player["gold"],
            "room": self.current_room_idx,
            "game_state": self.game_state,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("5-Room Dungeon Crawler")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # no-op, release, release
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, State: {info['game_state']}")
            if terminated:
                print(f"Episode finished. Final Score: {info['score']}")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(10) # Control game speed for human play
        
    env.close()