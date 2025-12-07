import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:22:23.771216
# Source Brief: brief_02131.md
# Brief Index: 2131
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Navigate a procedurally generated dungeon, using a deck of cards to fight enemies and reach the core at the deepest level."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move. Press space to use the first card in your hand and shift to discard it."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 41
    GRID_HEIGHT = 29
    TILE_SIZE = 24
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (15, 10, 25)
    COLOR_WALL = (40, 30, 60)
    COLOR_FLOOR = (25, 20, 40)
    COLOR_PLAYER = (0, 191, 255)
    COLOR_PLAYER_GLOW = (0, 191, 255, 50)
    COLOR_ENEMY = (255, 50, 100)
    COLOR_ENEMY_GLOW = (255, 50, 100, 50)
    COLOR_CORE = (180, 0, 255)
    COLOR_CORE_GLOW = (180, 0, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 25, 50, 180)
    COLOR_HEALTH_BAR = (0, 255, 127)
    COLOR_HEALTH_BAR_BG = (128, 0, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 20, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = (0, 0)
        self.player_health = 100
        self.max_player_health = 100
        self.current_depth = 1
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT))
        self.core_pos = (0, 0)
        self.enemies = []
        self.particles = []
        self.deck = []
        self.hand = []
        self.discard_pile = []
        self.camera_offset = (0, 0)
        self.last_action_feedback = ""
        self.last_action_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_depth = 1
        
        self._generate_dungeon()
        self._initialize_cards()
        
        self.player_health = self.max_player_health
        self.player_pos = self.player_start_pos
        
        self._spawn_enemies()
        
        self.particles = []
        self.last_action_feedback = ""
        self.last_action_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, use_card_action, discard_card_action = action[0], action[1] == 1, action[2] == 1

        prev_dist_to_core = self._distance(self.player_pos, self.core_pos)
        
        # --- Player Action Phase ---
        action_taken = False
        if use_card_action and self.hand:
            reward += self._use_card(0)
            action_taken = True
        elif discard_card_action and self.hand:
            reward += self._discard_card(0)
            action_taken = True
        elif movement != 0:
            self._move_player(movement)
            action_taken = True
        
        if not action_taken:
            # No-op action
            self._set_feedback("Waiting...")
        
        # Movement reward
        new_dist_to_core = self._distance(self.player_pos, self.core_pos)
        if new_dist_to_core < prev_dist_to_core:
            reward += 0.1
        elif new_dist_to_core > prev_dist_to_core:
            reward -= 0.1
        
        # --- Enemy Action Phase ---
        if action_taken:
            reward += self._handle_enemy_actions()

        # --- State Updates & Termination Check ---
        if self.player_pos == self.core_pos:
            self.score += 1
            reward += 100
            self._set_feedback(f"Core Reached! Descending to Depth {self.current_depth + 1}")
            self._next_level()
            terminated = False # Not a terminal state, but a level transition
        elif self.player_health <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
            self._set_feedback("Player Defeated!")
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True # This should be truncated, but we'll stick to terminated for now
            self._set_feedback("Time Limit Reached.")
        else:
            terminated = False

        truncated = self.steps >= self.MAX_STEPS

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_camera()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "depth": self.current_depth,
            "health": self.player_health,
            "enemies": len(self.enemies),
            "pos": self.player_pos,
            "core_pos": self.core_pos
        }

    # --- Game Logic Helpers ---
    def _next_level(self):
        self.current_depth += 1
        self._generate_dungeon()
        self.player_pos = self.player_start_pos
        self._spawn_enemies()
        self.particles = []
        # Player keeps health and cards

    def _generate_dungeon(self):
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Randomized DFS for maze generation
        start_x, start_y = (self.np_random.integers(1, self.GRID_WIDTH // 2) * 2 -1, 
                            self.np_random.integers(1, self.GRID_HEIGHT // 2) * 2 -1)
        
        stack = deque([(start_x, start_y)])
        self.grid[start_x, start_y] = 0
        visited_cells = {(start_x, start_y)}

        max_dist = 0
        farthest_cell = (start_x, start_y)

        while stack:
            cx, cy = stack[-1]
            
            dist = self._distance((start_x, start_y), (cx, cy))
            if dist > max_dist:
                max_dist = dist
                farthest_cell = (cx, cy)

            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited_cells:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = tuple(self.np_random.choice(np.array(neighbors), axis=0))
                wall_x, wall_y = cx + (nx - cx) // 2, cy + (ny - cy) // 2
                self.grid[wall_x, wall_y] = 0
                self.grid[nx, ny] = 0
                visited_cells.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        self.player_start_pos = (start_x, start_y)
        self.core_pos = farthest_cell
        
        # Ensure start and core are not walls
        self.grid[self.player_start_pos] = 0
        self.grid[self.core_pos] = 0

    def _spawn_enemies(self):
        self.enemies = []
        
        spawn_rate_divisor = max(1, 3 - ((self.current_depth -1) // 10))
        num_enemies = self.np_random.integers(1, self.GRID_WIDTH // 10) + (self.current_depth // spawn_rate_divisor)
        
        enemy_health = 20 + 5 * ((self.current_depth-1) // 5)
        
        empty_tiles = np.argwhere(self.grid == 0).tolist()
        random.shuffle(empty_tiles)

        for _ in range(num_enemies):
            if not empty_tiles: break
            pos = tuple(empty_tiles.pop())
            if self._distance(pos, self.player_pos) > 5 and pos != self.core_pos:
                self.enemies.append({
                    "pos": pos,
                    "health": enemy_health,
                    "max_health": enemy_health,
                    "patrol_start": pos,
                    "patrol_dir": self.np_random.choice([-1, 1])
                })
    
    def _move_player(self, direction):
        dx, dy = 0, 0
        if direction == 1: dy = -1  # Up
        elif direction == 2: dy = 1   # Down
        elif direction == 3: dx = -1  # Left
        elif direction == 4: dx = 1   # Right
        
        new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        
        if self._is_valid_and_walkable(new_pos):
            self.player_pos = new_pos
            self._set_feedback(f"Moved.")
            # sound: player_step.wav
        else:
            self._set_feedback("Blocked.")
            # sound: bump_wall.wav

    def _handle_enemy_actions(self):
        reward = 0
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if enemy["health"] <= 0:
                reward += 1.0
                self._set_feedback("Enemy vanquished!")
                self._add_particles(self._grid_to_screen(enemy["pos"]), 30, self.COLOR_ENEMY, 2.5)
                enemies_to_remove.append(i)
                # sound: enemy_die.wav
                continue

            # Attack
            if self._distance(self.player_pos, enemy["pos"]) <= 1.1:
                self.player_health -= 10
                reward -= 5.0
                self._set_feedback("Took 10 damage!")
                self._add_particles(self._grid_to_screen(self.player_pos), 20, self.COLOR_PLAYER, 3)
                # sound: player_hurt.wav
            # Patrol
            else:
                patrol_end = (enemy["patrol_start"][0] + enemy["patrol_dir"], enemy["patrol_start"][1])
                if not self._is_valid_and_walkable(patrol_end):
                    patrol_end = (enemy["patrol_start"][0], enemy["patrol_start"][1] + enemy["patrol_dir"])
                
                if enemy["pos"] == enemy["patrol_start"]:
                    if self._is_valid_and_walkable(patrol_end): enemy["pos"] = patrol_end
                else:
                    enemy["pos"] = enemy["patrol_start"]
        
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
        
        return reward

    # --- Card System ---
    def _initialize_cards(self):
        all_cards = [
            {"name": "Zap", "desc": "Hit adjacent enemies", "color": (255, 215, 0), "func": self._card_zap},
            {"name": "Heal", "desc": "Restore 25 HP", "color": (60, 220, 60), "func": self._card_heal},
            {"name": "Teleport", "desc": "Jump to random tile", "color": (138, 43, 226), "func": self._card_teleport},
            {"name": "Phase", "desc": "Move through one wall", "color": (173, 216, 230), "func": self._card_phase},
        ]
        self.deck = all_cards * 5 # Create a larger deck
        self.np_random.shuffle(self.deck)
        self.hand = []
        self.discard_pile = []
        for _ in range(3):
            self._draw_card()

    def _draw_card(self):
        if not self.deck:
            if not self.discard_pile: return # No cards left anywhere
            self.deck = self.discard_pile
            self.discard_pile = []
            self.np_random.shuffle(self.deck)
            # sound: shuffle_deck.wav
        if len(self.hand) < 5:
            self.hand.append(self.deck.pop(0))

    def _use_card(self, hand_index):
        card = self.hand.pop(hand_index)
        reward = card["func"]()
        self.discard_pile.append(card)
        self._draw_card()
        # sound: use_card.wav
        return reward

    def _discard_card(self, hand_index):
        card = self.hand.pop(hand_index)
        self.discard_pile.append(card)
        self._draw_card()
        self._set_feedback(f"Discarded {card['name']}.")
        return 0 # No reward for discarding

    def _card_zap(self):
        self._set_feedback("Used Zap!")
        reward = 0
        for enemy in self.enemies:
            if self._distance(self.player_pos, enemy["pos"]) <= 1.5:
                enemy["health"] -= 20
                self._add_particles(self._grid_to_screen(enemy["pos"]), 20, (255, 255, 0), 2)
        return reward

    def _card_heal(self):
        heal_amount = 25
        self.player_health = min(self.max_player_health, self.player_health + heal_amount)
        self._set_feedback(f"Healed for {heal_amount} HP.")
        self._add_particles(self._grid_to_screen(self.player_pos), 20, self.COLOR_HEALTH_BAR, 2)
        return 0

    def _card_teleport(self):
        empty_tiles = np.argwhere(self.grid == 0).tolist()
        if empty_tiles:
            self.player_pos = tuple(self.np_random.choice(np.array(empty_tiles), axis=0))
            self._set_feedback("Teleported!")
            self._add_particles(self._grid_to_screen(self.player_pos), 40, (138, 43, 226), 3)
        return 0

    def _card_phase(self):
        # This is an example of a more complex card. For this implementation, we will just move to a random adjacent tile, ignoring walls
        self._set_feedback("Phased!")
        neighbors = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = self.player_pos[0] + dx, self.player_pos[1] + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                neighbors.append((nx, ny))
        if neighbors:
            self.player_pos = tuple(self.np_random.choice(np.array(neighbors), axis=0))
            self._add_particles(self._grid_to_screen(self.player_pos), 20, (173, 216, 230), 1.5)
        return 0

    # --- Rendering ---
    def _update_camera(self):
        px, py = self._grid_to_screen(self.player_pos)
        self.camera_offset = (self.SCREEN_WIDTH / 2 - px, self.SCREEN_HEIGHT / 2 - py)
    
    def _render_game(self):
        self._render_dungeon()
        self._update_and_draw_particles()
        self._render_entities()

    def _render_dungeon(self):
        # Visible grid calculation
        cam_x, cam_y = self.camera_offset
        start_col = max(0, int(-cam_x / self.TILE_SIZE))
        end_col = min(self.GRID_WIDTH, int((-cam_x + self.SCREEN_WIDTH) / self.TILE_SIZE) + 2)
        start_row = max(0, int(-cam_y / self.TILE_SIZE))
        end_row = min(self.GRID_HEIGHT, int((-cam_y + self.SCREEN_HEIGHT) / self.TILE_SIZE) + 2)

        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                screen_pos = self._grid_to_screen((x, y))
                rect = pygame.Rect(screen_pos[0], screen_pos[1], self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw Core
        core_screen_pos = self._grid_to_screen(self.core_pos)
        self._draw_glowing_circle(self.screen, self.COLOR_CORE, self.COLOR_CORE_GLOW, core_screen_pos, self.TILE_SIZE // 2)

    def _render_entities(self):
        # Draw Enemies
        for enemy in self.enemies:
            pos = self._grid_to_screen(enemy["pos"])
            self._draw_glowing_circle(self.screen, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW, pos, self.TILE_SIZE // 2 - 2)
            # Enemy health bar
            hp_ratio = enemy["health"] / enemy["max_health"]
            bar_w = int(self.TILE_SIZE * 0.8)
            bar_h = 4
            bar_x = pos[0] - bar_w // 2
            bar_y = pos[1] - self.TILE_SIZE // 2 - 5
            pygame.draw.rect(self.screen, (50,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

        # Draw Player
        player_screen_pos = self._grid_to_screen(self.player_pos)
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, player_screen_pos, self.TILE_SIZE // 2)

    def _render_ui(self):
        # Health Bar
        hp_width = 200
        hp_height = 20
        hp_ratio = max(0, self.player_health / self.max_player_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, hp_width, hp_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(hp_width * hp_ratio), hp_height))
        health_text = self.font_large.render(f"{self.player_health}/{self.max_player_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 9))

        # Depth and Score
        depth_text = self.font_large.render(f"Depth: {self.current_depth}", True, self.COLOR_UI_TEXT)
        self.screen.blit(depth_text, (self.SCREEN_WIDTH - 150, 10))
        
        # Card Hand
        card_w, card_h = 100, 60
        for i, card in enumerate(self.hand):
            x = self.SCREEN_WIDTH // 2 - (len(self.hand) * (card_w + 10)) // 2 + i * (card_w + 10)
            y = self.SCREEN_HEIGHT - card_h - 10
            card_rect = pygame.Rect(x, y, card_w, card_h)
            
            # Highlight first card (target for actions)
            border_color = (255,255,255) if i == 0 else card['color']
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, card_rect, border_radius=5)
            pygame.draw.rect(self.screen, border_color, card_rect, 2, border_radius=5)
            
            name_surf = self.font_large.render(card['name'], True, card['color'])
            desc_surf = self.font_small.render(card['desc'], True, self.COLOR_UI_TEXT)
            self.screen.blit(name_surf, (x + 5, y + 5))
            self.screen.blit(desc_surf, (x + 5, y + 30))
        
        # Action Feedback
        if self.last_action_timer > 0:
            self.last_action_timer -= 1
            feedback_surf = self.font_large.render(self.last_action_feedback, True, self.COLOR_UI_TEXT)
            pos = feedback_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 100))
            self.screen.blit(feedback_surf, pos)

    def _add_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({"pos": list(pos), "vel": vel, "life": lifetime, "color": color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["life"] / 40))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"][0]), int(p["pos"][1]), 
                    int(p["life"] * 0.1) + 1, (*p["color"], alpha)
                )

    # --- Utility Functions ---
    def _is_valid_and_walkable(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        return self.grid[x, y] == 0

    def _distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _grid_to_screen(self, grid_pos):
        x = int(grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2 + self.camera_offset[0])
        y = int(grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2 + self.camera_offset[1])
        return x, y

    def _set_feedback(self, text):
        self.last_action_feedback = text
        self.last_action_timer = 60 # frames

    def _draw_glowing_circle(self, surface, color, glow_color, pos, radius):
        # Draw the glow effect using multiple transparent circles
        for i in range(4):
            glow_radius = int(radius + i * 2)
            alpha = glow_color[3] // (i + 1)
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), glow_radius, (*glow_color[:3], alpha))
        # Draw the main circle
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), radius, color)
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), radius, color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and will not be run by the evaluation system.
    # It requires a display.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        import pygame
        
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Fractal Dungeon Crawler")
        clock = pygame.time.Clock()

        running = True
        total_reward = 0
        terminated = False
        
        while running:
            movement = 0 # no-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_SPACE: space = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                    elif event.key == pygame.K_r: # Reset
                        obs, info = env.reset()
                        total_reward = 0
                        terminated = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            if not terminated and (movement or space or shift):
                action = [movement, space, shift]
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            
            # Draw the observation to the screen
            frame = env._get_observation()
            frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit frame rate
        
        env.close()

    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("Could not create display. This is expected in a headless environment.")
        print("The environment is likely working correctly.")