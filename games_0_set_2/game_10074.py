import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:53:39.762479
# Source Brief: brief_00074.md
# Brief Index: 74
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = random.randint(15, 30)
        self.radius = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.1
        return self.lifespan > 0 and self.radius > 0

    def draw(self, surface):
        if self.radius > 0:
            pos = (int(self.x), int(self.y))
            # Draw a filled circle with an anti-aliased border for a glowy effect
            try:
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), self.color)
                pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.radius), self.color)
            except OverflowError: # Can happen if particles fly way off-screen
                pass

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent blocks to create lines of three or more of the same color to score points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to select a block, then move to an adjacent block and press space again to swap. Press shift to deselect."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 5, 5
        self.BLOCK_SIZE = 60
        self.GRID_LINE_WIDTH = 2
        self.MOVE_LIMIT = 100
        self.MAX_EPISODE_STEPS = 1000

        self.GRID_WIDTH = self.GRID_COLS * self.BLOCK_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.BLOCK_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID_BG = (25, 30, 45)
        self.COLOR_GRID_LINES = (40, 50, 70)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 15)
        self.BLOCK_COLORS = [
            (227, 68, 68),   # Red
            (68, 227, 106),  # Green
            (68, 140, 227),  # Blue
            (227, 224, 68),  # Yellow
            (173, 68, 227),  # Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.moves_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_phase = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.particles = []
        self.animation_state = {}
        self.turn_reward = 0
        self.chain_level = 1

        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_pos = None
        self.moves_remaining = self.MOVE_LIMIT
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_phase = "INPUT"
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.animation_state = {}
        self.turn_reward = 0
        self.chain_level = 1

        while True:
            self.grid = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            if not self._find_matches(self.grid):
                break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        truncated = False

        self._update_particles()
        
        if self.game_phase != "INPUT":
            self._update_animations()
        else:
            if self.game_over:
                # If game is over, only a reset can change things.
                pass
            else:
                reward_from_input, _ = self._handle_input(action)
                reward += reward_from_input

        if self.game_phase == "INPUT" and self.turn_reward > 0:
            reward += self.turn_reward
            self.turn_reward = 0
            
        if not self.game_over:
            is_cleared = np.all(self.grid == 0)
            if is_cleared:
                reward += 100
                self.score += 100
                terminated = True
                self.game_over = True
                self.game_phase = "WIN"
            elif self.moves_remaining <= 0:
                reward -= 100
                terminated = True
                self.game_over = True
                self.game_phase = "LOSE"

        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Handle Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # --- Handle Selection/Deselection ---
        if shift_pressed and self.selected_pos:
            self.selected_pos = None
        
        if space_pressed:
            if not self.selected_pos:
                self.selected_pos = list(self.cursor_pos)
            else:
                # Attempt a swap
                if self._is_adjacent(self.selected_pos, self.cursor_pos):
                    self.moves_remaining -= 1
                    
                    p1, p2 = self.selected_pos, self.cursor_pos
                    v1 = self.grid[p1[1], p1[0]]
                    v2 = self.grid[p2[1], p2[0]]

                    # Temporarily swap to check for matches
                    self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = v2, v1
                    
                    if self._find_matches(self.grid):
                        # Valid swap, start animation
                        self.animation_state = {
                            "type": "SWAP",
                            "p1": p1, "p2": p2,
                            "v1": v2, "v2": v1,
                            "progress": 0, "duration": 10
                        }
                        self.game_phase = "ANIMATING"
                        self.chain_level = 1
                    else:
                        # Invalid swap, swap back
                        self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = v1, v2
                        return -0.1, False # Return small penalty
                else:
                    # Not adjacent, just deselect
                    pass
                self.selected_pos = None
        
        return 0, False

    def _update_animations(self):
        if not self.animation_state:
            self.game_phase = "INPUT"
            return

        self.animation_state["progress"] += 1
        if self.animation_state["progress"] >= self.animation_state["duration"]:
            phase_type = self.animation_state["type"]
            self.animation_state = {}

            if phase_type == "SWAP":
                self._start_match_check()
            elif phase_type == "CLEAR":
                self._start_gravity()
            elif phase_type == "FALL":
                self._start_refill()
            elif phase_type == "REFILL":
                self._start_match_check()
    
    def _start_match_check(self):
        matches = self._find_matches(self.grid)
        if matches:
            blocks_cleared = len(matches)
            reward_per_block = 1
            self.turn_reward += blocks_cleared * reward_per_block * self.chain_level
            self.score += blocks_cleared * reward_per_block * self.chain_level
            
            for r, c in matches:
                self._create_particles(c, r, self.grid[r, c])
                self.grid[r, c] = 0 # Mark for clearing
            
            self.animation_state = {
                "type": "CLEAR",
                "progress": 0, "duration": 8
            }
            self.game_phase = "ANIMATING"
            self.chain_level += 1
        else:
            self.game_phase = "INPUT"
            self.chain_level = 1

    def _start_gravity(self):
        moved_blocks = []
        for c in range(self.GRID_COLS):
            write_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    val = self.grid[r, c]
                    if r != write_row:
                        moved_blocks.append({
                            "from": (c, r), "to": (c, write_row), "val": val
                        })
                    self.grid[r,c] = 0
                    self.grid[write_row, c] = val
                    write_row -= 1

        if moved_blocks:
            self.animation_state = {
                "type": "FALL",
                "blocks": moved_blocks,
                "progress": 0, "duration": 10
            }
            self.game_phase = "ANIMATING"
        else:
            self._start_refill()

    def _start_refill(self):
        new_blocks = []
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[r, c] == 0:
                    val = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
                    self.grid[r, c] = val
                    new_blocks.append({"pos": (c, r), "val": val})
        
        if new_blocks:
            self.animation_state = {
                "type": "REFILL",
                "blocks": new_blocks,
                "progress": 0, "duration": 10
            }
            self.game_phase = "ANIMATING"
        else:
            self._start_match_check()

    def _find_matches(self, grid):
        matches = set()
        # Check rows
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                val = grid[r, c]
                if val != 0 and val == grid[r, c+1] and val == grid[r, c+2]:
                    for i in range(3): matches.add((r, c+i))
                    for i in range(c + 3, self.GRID_COLS):
                        if grid[r, i] == val: matches.add((r, i))
                        else: break
        # Check columns
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                val = grid[r, c]
                if val != 0 and val == grid[r+1, c] and val == grid[r+2, c]:
                    for i in range(3): matches.add((r+i, c))
                    for i in range(r + 3, self.GRID_ROWS):
                        if grid[i, c] == val: matches.add((i, c))
                        else: break
        return matches
    
    def _is_adjacent(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1
    
    def _create_particles(self, c, r, val):
        center_x = self.GRID_OFFSET_X + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        center_y = self.GRID_OFFSET_Y + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[val - 1]
        for _ in range(20):
            self.particles.append(Particle(center_x, center_y, color))
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _render_game(self):
        # Draw grid background
        grid_rect = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw blocks
        temp_grid = np.copy(self.grid)
        anim = self.animation_state
        
        if anim and anim.get("type") in ["SWAP", "FALL", "REFILL"]:
            if anim["type"] == "SWAP":
                p1, p2, v1, v2 = anim["p1"], anim["p2"], anim["v1"], anim["v2"]
                temp_grid[p1[1], p1[0]] = 0
                temp_grid[p2[1], p2[0]] = 0
                self._draw_animated_block(p1, p2, v1, anim)
                self._draw_animated_block(p2, p1, v2, anim)
            elif anim["type"] in ["FALL", "REFILL"]:
                for block_anim in anim["blocks"]:
                    if anim["type"] == "FALL":
                        c, r = block_anim["from"]
                        temp_grid[r, c] = 0
                        self._draw_animated_block(block_anim["from"], block_anim["to"], block_anim["val"], anim)
                    else: # REFILL
                        c, r = block_anim["pos"]
                        temp_grid[r, c] = 0
                        start_pos = (c, r - 1)
                        self._draw_animated_block(start_pos, block_anim["pos"], block_anim["val"], anim)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                val = temp_grid[r, c]
                if val != 0:
                    scale = 1.0
                    if anim and anim.get("type") == "CLEAR" and self.grid[r,c] == 0:
                        scale = 1.0 - anim["progress"] / anim["duration"]
                    self._draw_block(c, r, val, scale)

        # Draw grid lines
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT), self.GRID_LINE_WIDTH)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw cursor and selection
        if not self.game_over:
            self._draw_cursor(self.cursor_pos[0], self.cursor_pos[1])
            if self.selected_pos:
                self._draw_selection(self.selected_pos[0], self.selected_pos[1])

    def _draw_block(self, c, r, val, scale=1.0):
        color = self.BLOCK_COLORS[val - 1]
        size = self.BLOCK_SIZE * scale
        margin = (self.BLOCK_SIZE - size) / 2
        rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.BLOCK_SIZE + margin,
            self.GRID_OFFSET_Y + r * self.BLOCK_SIZE + margin,
            size, size
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=int(self.BLOCK_SIZE * 0.2))

    def _draw_animated_block(self, from_pos, to_pos, val, anim):
        t = anim["progress"] / anim["duration"]
        t = 1 - (1 - t) * (1 - t) # Ease-out quadratic
        
        start_x = self.GRID_OFFSET_X + from_pos[0] * self.BLOCK_SIZE
        start_y = self.GRID_OFFSET_Y + from_pos[1] * self.BLOCK_SIZE
        end_x = self.GRID_OFFSET_X + to_pos[0] * self.BLOCK_SIZE
        end_y = self.GRID_OFFSET_Y + to_pos[1] * self.BLOCK_SIZE
        
        draw_x = start_x + (end_x - start_x) * t
        draw_y = start_y + (end_y - start_y) * t
        
        rect = pygame.Rect(draw_x, draw_y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.BLOCK_COLORS[val - 1], rect, border_radius=int(self.BLOCK_SIZE * 0.2))

    def _draw_cursor(self, c, r):
        rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.BLOCK_SIZE,
            self.GRID_OFFSET_Y + r * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 4, border_radius=int(self.BLOCK_SIZE * 0.2))

    def _draw_selection(self, c, r):
        pulse = abs(math.sin(self.steps * 0.2))
        alpha = 50 + pulse * 50
        color = (*self.COLOR_CURSOR, int(alpha))
        
        surf = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(surf, color, surf.get_rect(), border_radius=int(self.BLOCK_SIZE * 0.2))
        
        self.screen.blit(surf, (self.GRID_OFFSET_X + c * self.BLOCK_SIZE, self.GRID_OFFSET_Y + r * self.BLOCK_SIZE))

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_remaining}"
        score_text = f"Score: {self.score}"
        
        self._draw_text(moves_text, (20, 20), self.font_main)
        score_pos = (self.WIDTH - self.font_main.size(score_text)[0] - 20, 20)
        self._draw_text(score_text, score_pos, self.font_main)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.game_phase == "WIN" else "GAME OVER"
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2), self.font_large, center=True)

    def _draw_text(self, text, pos, font, center=False):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, self.COLOR_TEXT)
        
        if center:
            text_rect = text_surf.get_rect(center=pos)
            shadow_rect = shadow_surf.get_rect(center=(pos[0]+2, pos[1]+2))
        else:
            text_rect = text_surf.get_rect(topleft=pos)
            shadow_rect = shadow_surf.get_rect(topleft=(pos[0]+2, pos[1]+2))
            
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Switch to a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    pygame.quit() # Quit the dummy instance
    pygame.init() # Re-init with the real driver
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    # Re-render the initial state on the new screen
    frame = np.transpose(obs, (1, 0, 2))
    surf = pygame.surfarray.make_surface(frame)
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    
    while running:
        # --- Event Handling ---
        movement = 0 # Reset movement action each frame
        space_pressed_this_frame = False
        shift_pressed_this_frame = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    space_pressed_this_frame = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_pressed_this_frame = True
        
        # --- Movement Handling (key held down) ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # --- Step Environment ---
        # The env expects held state, but for manual play, single presses are better
        # We simulate a "held" state for just one frame on a key press event
        action = [movement, 1 if space_pressed_this_frame else 0, 1 if shift_pressed_this_frame else 0]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
        
        if terminated or truncated:
            print("Game Over!")
            # The game will now show the final screen until 'R' is pressed.
            
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Slower tick for manual play to register single presses

    env.close()