
# Generated: 2025-08-28T02:16:28.289315
# Source Brief: brief_04392.md
# Brief Index: 4392

        
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
        "Controls: Use Space/Shift to cycle which block is selected. Use arrow keys to push the selected block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Push all colored blocks onto their matching targets before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = (10, 10)
        self.NUM_BLOCKS = 15
        self.MAX_MOVES = 20

        # Visuals
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = self.TILE_WIDTH // 2
        self.BLOCK_HEIGHT = 20
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        self.COLOR_BG = pygame.Color(30, 40, 50)
        self.COLOR_GRID = pygame.Color(60, 75, 90)
        self.BLOCK_COLORS_HEX = [0xFF6347, 0x4682B4, 0x32CD32, 0xFFD700, 0x9370DB]
        self.BLOCK_COLORS = [pygame.Color(c) for c in self.BLOCK_COLORS_HEX]
        self.TARGET_COLORS = [c.lerp(self.COLOR_GRID, 0.5) for c in self.BLOCK_COLORS]
        self.COLOR_WHITE = pygame.Color(255, 255, 255)
        self.COLOR_TEXT = pygame.Color(220, 220, 220)
        self.COLOR_WIN = pygame.Color(144, 238, 144)
        self.COLOR_LOSE = pygame.Color(255, 99, 71)

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
        self.font_main = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)
        
        # --- Game State ---
        self.blocks = []
        self.board = np.full(self.GRID_SIZE, -1, dtype=int)
        self.moves_remaining = 0
        self.selected_block_idx = 0
        self.game_over = False
        self.win_status = False
        self.steps = 0
        self.particles = []
        self.score = 0
        
        # Initialize state variables
        self.reset()
    
    def _iso_to_screen(self, gx, gy):
        x = self.ORIGIN_X + (gx - gy) * self.TILE_WIDTH / 2
        y = self.ORIGIN_Y + (gx + gy) * self.TILE_HEIGHT / 2
        return int(x), int(y)

    def _place_entities(self):
        self.blocks = []
        self.board.fill(-1)
        
        possible_coords = [(x, y) for x in range(self.GRID_SIZE[0]) for y in range(self.GRID_SIZE[1])]
        self.np_random.shuffle(possible_coords)
        
        used_coords = set()

        for i in range(self.NUM_BLOCKS):
            color_idx = i % len(self.BLOCK_COLORS)
            
            # Place target
            target_pos = possible_coords.pop()
            used_coords.add(target_pos)

            # Place block
            block_pos = possible_coords.pop()
            used_coords.add(block_pos)
            
            block = {
                "id": i,
                "pos": block_pos,
                "target_pos": target_pos,
                "color": self.BLOCK_COLORS[color_idx],
                "target_color": self.TARGET_COLORS[color_idx],
                "y_order": block_pos[0] + block_pos[1]
            }
            self.blocks.append(block)
            self.board[block_pos] = i

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_remaining = self.MAX_MOVES
        self.selected_block_idx = 0
        self.game_over = False
        self.win_status = False
        self.steps = 0
        self.particles = []
        self.score = 0
        
        self._place_entities()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Action priority: Push > Select
        if movement > 0: # Push action
            self.moves_remaining -= 1
            
            # Get push direction
            # 1=up (iso up-left), 2=down (iso down-right), 3=left (iso down-left), 4=right (iso up-right)
            dirs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            dx, dy = dirs[movement]
            
            block = self.blocks[self.selected_block_idx]
            old_pos = block['pos']
            
            # Calculate distance for reward
            old_dist = abs(old_pos[0] - block['target_pos'][0]) + abs(old_pos[1] - block['target_pos'][1])
            
            # Find new position
            cx, cy = old_pos
            nx, ny = cx + dx, cy + dy
            while (0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1] and self.board[nx, ny] == -1):
                cx, cy = nx, ny
                nx, ny = cx + dx, cy + dy
            
            new_pos = (cx, cy)
            
            if new_pos != old_pos:
                # Update board and block state
                self.board[old_pos] = -1
                self.board[new_pos] = block['id']
                block['pos'] = new_pos
                block['y_order'] = new_pos[0] + new_pos[1]
                
                # Add particles for visual feedback
                self._create_push_particles(old_pos, new_pos, block['color'])
                # sfx: block slide
            
            # Calculate reward
            new_dist = abs(new_pos[0] - block['target_pos'][0]) + abs(new_pos[1] - block['target_pos'][1])
            
            if new_dist < old_dist: reward += 1.0
            elif new_dist > old_dist: reward -= 1.0

            was_on_target = old_pos == block['target_pos']
            is_on_target = new_pos == block['target_pos']

            if is_on_target and not was_on_target: reward += 5.0 # sfx: block on target
            if was_on_target and not is_on_target: reward -= 5.0

        elif space_held and not shift_held: # Select next
            self.selected_block_idx = (self.selected_block_idx + 1) % self.NUM_BLOCKS
            # sfx: selection change
        elif shift_held and not space_held: # Select previous
            self.selected_block_idx = (self.selected_block_idx - 1 + self.NUM_BLOCKS) % self.NUM_BLOCKS
            # sfx: selection change

        self.score += reward
        
        # Check termination conditions
        win_condition_met = all(b['pos'] == b['target_pos'] for b in self.blocks)
        
        if win_condition_met:
            self.game_over = True
            self.win_status = True
            reward += 100.0 # sfx: win fanfare
        elif self.moves_remaining <= 0:
            self.game_over = True
            self.win_status = False
            reward -= 100.0 # sfx: lose sound

        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
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
            "moves_remaining": self.moves_remaining,
        }

    def _draw_iso_cube(self, surface, gx, gy, color, height):
        x, y = self._iso_to_screen(gx, gy)
        w, h = self.TILE_WIDTH, self.TILE_HEIGHT
        
        top_face = [
            (x, y - h / 2),
            (x + w / 2, y),
            (x, y + h / 2),
            (x - w / 2, y),
        ]
        left_face = [
            (x - w / 2, y),
            (x, y + h / 2),
            (x, y + h / 2 + height),
            (x - w / 2, y + height),
        ]
        right_face = [
            (x + w / 2, y),
            (x, y + h / 2),
            (x, y + h / 2 + height),
            (x + w / 2, y + height),
        ]
        
        c_top = color
        c_left = color.lerp((0,0,0), 0.3)
        c_right = color.lerp((0,0,0), 0.6)

        pygame.gfxdraw.filled_polygon(surface, top_face, c_top)
        pygame.gfxdraw.aapolygon(surface, top_face, c_top)
        pygame.gfxdraw.filled_polygon(surface, left_face, c_left)
        pygame.gfxdraw.aapolygon(surface, left_face, c_left)
        pygame.gfxdraw.filled_polygon(surface, right_face, c_right)
        pygame.gfxdraw.aapolygon(surface, right_face, c_right)

    def _draw_iso_rect(self, surface, gx, gy, color):
        x, y = self._iso_to_screen(gx, gy)
        w, h = self.TILE_WIDTH, self.TILE_HEIGHT
        points = [
            (x, y - h / 2),
            (x + w / 2, y),
            (x, y + h / 2),
            (x - w / 2, y),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color.lerp((255,255,255), 0.2))

    def _create_push_particles(self, start_pos, end_pos, color):
        start_screen = self._iso_to_screen(start_pos[0], start_pos[1])
        end_screen = self._iso_to_screen(end_pos[0], end_pos[1])
        
        dist = math.hypot(end_screen[0] - start_screen[0], end_screen[1] - start_screen[1])
        if dist == 0: return

        num_particles = int(dist / 5)
        for i in range(num_particles):
            t = i / max(1, num_particles - 1)
            pos = [start_screen[0] * (1 - t) + end_screen[0] * t, start_screen[1] * (1 - t) + end_screen[1] * t]
            self.particles.append({
                "pos": pos,
                "life": self.np_random.integers(10, 20),
                "max_life": 20,
                "color": color,
                "vel": [self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)]
            })

    def _render_game(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                p_color = (*p['color'][:3], alpha)
                radius = int(3 * (p['life'] / p['max_life']))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p_color)

        draw_list = []
        for block in self.blocks:
            draw_list.append({'type': 'target', 'pos': block['target_pos'], 'color': block['target_color'], 'y_order': block['target_pos'][0] + block['target_pos'][1]})
        
        for block in self.blocks:
            draw_list.append({'type': 'block', 'data': block, 'y_order': block['y_order']})
        
        draw_list.sort(key=lambda item: item['y_order'])

        for r in range(self.GRID_SIZE[0] + 1):
            p1 = self._iso_to_screen(r, -0.5)
            p2 = self._iso_to_screen(r, self.GRID_SIZE[1] - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.GRID_SIZE[1] + 1):
            p1 = self._iso_to_screen(-0.5, c)
            p2 = self._iso_to_screen(self.GRID_SIZE[0] - 0.5, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

        for item in draw_list:
            if item['type'] == 'target':
                self._draw_iso_rect(self.screen, item['pos'][0], item['pos'][1], item['color'])
            elif item['type'] == 'block':
                block = item['data']
                self._draw_iso_cube(self.screen, block['pos'][0], block['pos'][1], block['color'], self.BLOCK_HEIGHT)

        if not self.game_over:
            selected_block = self.blocks[self.selected_block_idx]
            sx, sy = self._iso_to_screen(selected_block['pos'][0], selected_block['pos'][1])
            sy += self.BLOCK_HEIGHT // 2 + 5
            
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            radius = int(10 + pulse * 4)
            alpha = int(100 + pulse * 100)
            
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, (255, 255, 255, alpha))
            pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, (255, 255, 255, alpha))

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_remaining}/{self.MAX_MOVES}"
        text_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_status else "OUT OF MOVES"
            color = self.COLOR_WIN if self.win_status else self.COLOR_LOSE
            
            msg_surf = self.font_big.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Isometric Block Pusher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        movement, space, shift = 0, 0, 0
        action_taken_this_frame = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken_this_frame = True
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    action_taken_this_frame = False
                elif event.key == pygame.K_q: running = False

        if action_taken_this_frame and not terminated:
            action = np.array([movement, space, shift])
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Moves: {info['moves_remaining']}, Terminated: {terminated}")
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()