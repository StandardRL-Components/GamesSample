import gymnasium as gym
import os
import pygame
import numpy as np
import math
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
from collections import deque
import os
import pygame


# Set the SDL video driver to dummy to run headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Place magical bubbles to trap incoming enemies before they reach your base at the bottom of the screen."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to place a bubble. Press shift to cycle through bubble types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 20, 15
    TILE_SIZE = 24
    GAME_AREA_W, GAME_AREA_H = GRID_W * TILE_SIZE, GRID_H * TILE_SIZE
    OFFSET_X = (SCREEN_W - GAME_AREA_W) // 2
    OFFSET_Y = (SCREEN_H - GAME_AREA_H) // 2

    # Colors
    COLOR_BG = (12, 18, 33)
    COLOR_GRID = (30, 40, 60)
    COLOR_BASE = (180, 50, 50)
    COLOR_BASE_GLOW = (255, 80, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 100, 100)
    
    BUBBLE_TYPES = {
        'green': {'color': (80, 220, 100), 'timer': 5},
        'red': {'color': (255, 100, 80), 'timer': 3},
        'blue': {'color': (80, 150, 255), 'timer': 7}
    }
    BUBBLE_TYPE_ORDER = ['green', 'red', 'blue']
    
    ENEMY_TYPES = {
        'grunt': {'color': (200, 80, 180), 'speed_mod': 1.0},
        'fast': {'color': (240, 80, 80), 'speed_mod': 1.5},
        'slow': {'color': (80, 200, 240), 'speed_mod': 0.75}
    }

    WIN_TURN = 50
    MAX_STEPS = 1000
    GAME_LOGIC_TICK_RATE = 5  # Update game state every 5 steps (frames)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.turn = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H - 2]
        self.base_pos = [self.GRID_W // 2, self.GRID_H - 1]
        
        self.enemies = []
        self.bubbles = []
        
        self.selected_bubble_idx = 0
        self.last_shift_press = False
        self.last_space_press = False
        
        self.enemy_spawn_count = 1
        self.base_enemy_speed = 1.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        # --- 1. Handle Player Input (every frame) ---
        movement, space_btn, shift_btn = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Cycle bubble type (on press)
        if shift_btn and not self.last_shift_press:
            self.selected_bubble_idx = (self.selected_bubble_idx + 1) % len(self.BUBBLE_TYPE_ORDER)
        self.last_shift_press = shift_btn

        # Place bubble (on press)
        if space_btn and not self.last_space_press:
            if self._is_valid_placement(self.cursor_pos):
                bubble_type_name = self.BUBBLE_TYPE_ORDER[self.selected_bubble_idx]
                bubble_info = self.BUBBLE_TYPES[bubble_type_name]
                self.bubbles.append({
                    'pos': list(self.cursor_pos),
                    'type': bubble_type_name,
                    'timer': bubble_info['timer'],
                    'trapped_enemy': None,
                    'creation_step': self.steps,
                })
        self.last_space_press = space_btn
        
        # --- 2. Advance Game State on Tick ---
        if self.steps % self.GAME_LOGIC_TICK_RATE == 0:
            self.turn += 1

            # --- 3. Update Bubbles ---
            bubbles_to_pop = []
            for i, bubble in enumerate(self.bubbles):
                bubble['timer'] -= 1
                if bubble['timer'] <= 0:
                    bubbles_to_pop.append(i)
                    if bubble['trapped_enemy'] is not None:
                        reward += 5.0  # Defeated an enemy
                        self.score += 50
                        bubble['trapped_enemy']['defeated'] = True
            
            self.bubbles = [b for i, b in enumerate(self.bubbles) if i not in bubbles_to_pop]
            self.enemies = [e for e in self.enemies if not e.get('defeated', False)]

            # --- 4. Update Enemies ---
            occupied_by_bubble = {tuple(b['pos']) for b in self.bubbles}
            for enemy in self.enemies:
                if self.game_over: break

                if enemy['trapped_in'] is not None:
                    if not any(b for b in self.bubbles if tuple(b['pos']) == tuple(enemy['pos'])):
                        enemy['trapped_in'] = None
                    continue

                old_dist = self._manhattan_distance(enemy['pos'], self.base_pos)
                next_pos = self._find_path(enemy['pos'], occupied_by_bubble)
                enemy['pos'] = next_pos
                new_dist = self._manhattan_distance(enemy['pos'], self.base_pos)
                if new_dist < old_dist:
                    reward -= 0.1

                if tuple(enemy['pos']) == tuple(self.base_pos):
                    self.game_over = True
                    reward -= 10.0
                    break

                for bubble in self.bubbles:
                    if tuple(bubble['pos']) == tuple(enemy['pos']) and bubble['trapped_enemy'] is None:
                        enemy['trapped_in'] = bubble
                        bubble['trapped_enemy'] = enemy
                        reward += 1.0
                        self.score += 10
                        break
            
            if self.game_over:
                self.enemies = [e for e in self.enemies if tuple(e['pos']) != tuple(self.base_pos)]

            # --- 5. Spawn New Enemies ---
            if not self.game_over and self.turn <= self.WIN_TURN:
                if self.turn > 0: # No spawn on first turn
                    for _ in range(self.enemy_spawn_count):
                        spawn_pos = [self.np_random.integers(0, self.GRID_W), 0]
                        if not any(e['pos'] == spawn_pos for e in self.enemies):
                            enemy_type = self.np_random.choice(list(self.ENEMY_TYPES.keys()))
                            self.enemies.append({
                                'pos': spawn_pos,
                                'type': enemy_type,
                                'speed_mod': self.ENEMY_TYPES[enemy_type]['speed_mod'],
                                'trapped_in': None,
                                'creation_step': self.steps,
                            })

            # --- 6. Update Difficulty ---
            if self.turn > 0 and self.turn % 5 == 0:
                self.enemy_spawn_count = min(5, self.enemy_spawn_count + 1)
            if self.turn > 0 and self.turn % 10 == 0:
                self.base_enemy_speed = min(2.0, self.base_enemy_speed + 0.1)

            # --- 7. Check Win Condition ---
            if not self.game_over and self.turn >= self.WIN_TURN:
                self.win = True
                self.game_over = True
                reward += 100.0
                self.score += 1000
        
        # --- 8. Check Termination Conditions ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W + 1):
            start = (self.OFFSET_X + x * self.TILE_SIZE, self.OFFSET_Y)
            end = (self.OFFSET_X + x * self.TILE_SIZE, self.OFFSET_Y + self.GAME_AREA_H)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_H + 1):
            start = (self.OFFSET_X, self.OFFSET_Y + y * self.TILE_SIZE)
            end = (self.OFFSET_X + self.GAME_AREA_W, self.OFFSET_Y + y * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw base
        base_rect = self._get_tile_rect(self.base_pos)
        glow_rect = base_rect.inflate(8, 8)
        glow_alpha = 100 + 50 * math.sin(self.steps * 0.1)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_BASE_GLOW, glow_alpha), s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)
        
        # Draw bubbles
        for bubble in self.bubbles:
            self._draw_bubble(bubble)

        # Draw enemies
        for enemy in self.enemies:
            self._draw_enemy(enemy)

        # Draw cursor
        is_valid = self._is_valid_placement(self.cursor_pos)
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        cursor_rect = self._get_tile_rect(self.cursor_pos)
        alpha = 150 + 100 * math.sin(self.steps * 0.2)
        pygame.gfxdraw.rectangle(self.screen, cursor_rect, (*cursor_color, alpha))
        pygame.gfxdraw.rectangle(self.screen, cursor_rect.inflate(-2, -2), (*cursor_color, alpha))

    def _draw_bubble(self, bubble):
        rect = self._get_tile_rect(bubble['pos'])
        bubble_type = bubble['type']
        color = self.BUBBLE_TYPES[bubble_type]['color']
        
        wobble = math.sin(bubble['creation_step'] * 0.5 + self.steps * 0.1) * 2
        center = (rect.centerx + wobble, rect.centery)
        radius = self.TILE_SIZE // 2 - 2

        glow_alpha = 80 + 40 * math.sin(bubble['creation_step'] + self.steps * 0.15)
        pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), int(radius + 3), (*color, glow_alpha))

        pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), int(radius), (255, 255, 255, 180))

        shine_progress = ((self.steps + bubble['creation_step'] * 5) % 80) / 80.0
        shine_y = center[1] - radius + (2 * radius * shine_progress)
        shine_x_offset = math.sqrt(max(0, radius**2 - (shine_y - center[1])**2))
        start_pos = (center[0] - shine_x_offset, shine_y)
        end_pos = (center[0] + shine_x_offset, shine_y)
        pygame.draw.line(self.screen, (255, 255, 255, 100), start_pos, end_pos, 2)
    
    def _draw_enemy(self, enemy):
        rect = self._get_tile_rect(enemy['pos'])
        color = self.ENEMY_TYPES[enemy['type']]['color']

        bob = math.sin(enemy['creation_step'] * 0.3 + self.steps * 0.15) * 2
        
        if enemy['trapped_in']:
            pygame.draw.rect(self.screen, color, rect.inflate(-12, -12), border_radius=4)
        else:
            body_rect = rect.inflate(-8, -8)
            body_rect.y += bob
            pygame.draw.rect(self.screen, color, body_rect, border_radius=4)
            eye_y = body_rect.centery - 2
            eye_l_x = body_rect.centerx - 4
            eye_r_x = body_rect.centerx + 4
            pygame.draw.rect(self.screen, (255, 255, 255), (eye_l_x-1, eye_y-1, 3, 3))
            pygame.draw.rect(self.screen, (255, 255, 255), (eye_r_x-1, eye_y-1, 3, 3))
            pygame.draw.rect(self.screen, (0,0,0), (eye_l_x, eye_y, 1, 1))
            pygame.draw.rect(self.screen, (0,0,0), (eye_r_x, eye_y, 1, 1))

    def _render_ui(self):
        turn_text = self.font_large.render(f"TURN: {self.turn}/{self.WIN_TURN}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, (15, 15))

        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 45))

        ui_text = self.font_small.render("BUBBLE TYPE", True, self.COLOR_TEXT)
        self.screen.blit(ui_text, (self.SCREEN_W - 15 - ui_text.get_width(), 15))
        
        bubble_type_name = self.BUBBLE_TYPE_ORDER[self.selected_bubble_idx]
        bubble_info = self.BUBBLE_TYPES[bubble_type_name]
        color = bubble_info['color']
        pos = (self.SCREEN_W - 45, 55)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, (255, 255, 255, 180))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "turn": self.turn}

    def _get_tile_rect(self, pos):
        return pygame.Rect(
            self.OFFSET_X + pos[0] * self.TILE_SIZE,
            self.OFFSET_Y + pos[1] * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE
        )

    def _is_valid_placement(self, pos):
        if tuple(pos) == tuple(self.base_pos): return False
        if any(tuple(b['pos']) == tuple(pos) for b in self.bubbles): return False
        return True

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _find_path(self, start_pos, occupied_tiles):
        neighbors = [
            (start_pos[0], start_pos[1] - 1),  # Up
            (start_pos[0], start_pos[1] + 1),  # Down
            (start_pos[0] - 1, start_pos[1]),  # Left
            (start_pos[0] + 1, start_pos[1]),  # Right
        ]
        
        valid_neighbors = []
        for n_pos in neighbors:
            if not (0 <= n_pos[0] < self.GRID_W and 0 <= n_pos[1] < self.GRID_H):
                continue
            if n_pos in occupied_tiles:
                continue
            valid_neighbors.append(n_pos)
        
        if not valid_neighbors:
            return list(start_pos)

        valid_neighbors.sort(key=lambda p: (self._manhattan_distance(p, self.base_pos), p[1], p[0]))
        
        return list(valid_neighbors[0])

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    pygame.display.set_caption("Bubble Bobble Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Control game speed for manual play

    pygame.quit()