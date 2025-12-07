import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:50:19.556604
# Source Brief: brief_02399.md
# Brief Index: 2399
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Use a robotic arm to catch falling blocks and place them to clear lines in this fast-paced puzzle game."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the arm. Press space to grab or release a block. "
        "Press shift to perform a quick drop (unlocked after 10 lines)."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2
    MAX_STEPS = 5000
    ARM_SPEED = 1 # grid cells per step

    # --- Colors ---
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 55)
    COLOR_ARM = (180, 190, 200)
    COLOR_ARM_GLOW = (210, 220, 230)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (80, 255, 255),  # Cyan
        (255, 80, 255),  # Magenta
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_gameover = pygame.font.SysFont("Consolas", 72, bold=True)
        
        self.grid = None
        self.arm_pos = None
        self.arm_holding_block = None
        self.falling_block = None
        self.next_block = None
        self.fall_speed_base = 0.03 # cells per step
        self.fall_speed_current = 0.0
        self.lines_cleared = 0
        self.unlocked_maneuvers = None
        self.particles = None
        self.lines_to_clear = None
        self.clear_line_timer = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.reward_this_step = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.arm_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.arm_holding_block = None
        self.fall_speed_current = self.fall_speed_base
        self.lines_cleared = 0
        self.unlocked_maneuvers = set()
        
        self.particles = []
        self.lines_to_clear = []
        self.clear_line_timer = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self._spawn_block(is_initial=True) # Spawn first block
        self._spawn_block() # Queue up the next block
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1
        
        self._update_particles()
        
        if self.clear_line_timer > 0:
            self.clear_line_timer -= 1
            if self.clear_line_timer == 0:
                self._execute_line_clear()
        else:
            self._handle_input(action)
            self._update_game_state()

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        reward = self.reward_this_step
        if self.game_over:
            reward -= 10
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held

        # --- Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -self.ARM_SPEED
        elif movement == 2: dy = self.ARM_SPEED
        elif movement == 3: dx = -self.ARM_SPEED
        elif movement == 4: dx = self.ARM_SPEED
        
        if dx != 0 or dy != 0:
            self.arm_pos[0] = np.clip(self.arm_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            self.arm_pos[1] = np.clip(self.arm_pos[1] + dy, 0, self.GRID_HEIGHT - 1)

        # --- Grab/Release (Space) ---
        if space_press:
            if self.arm_holding_block:
                # Release
                # sfx: block_release
                block = self.arm_holding_block
                block['pos'] = [float(self.arm_pos[0]), float(self.arm_pos[1])]
                self.falling_block = block
                self.arm_holding_block = None
            else:
                # Grab
                if self.falling_block:
                    arm_world_pos = self._grid_to_world(self.arm_pos)
                    block_world_pos = self._grid_to_world(self.falling_block['pos'])
                    if math.dist(arm_world_pos, block_world_pos) < self.CELL_SIZE * 1.5:
                        # sfx: block_grab
                        self.arm_holding_block = self.falling_block
                        self.falling_block = None
                        self.arm_holding_block['was_caught'] = True
                        self.reward_this_step += 0.1
        
        # --- Quick Drop (Shift) ---
        if shift_press and self.arm_holding_block and 'quick_drop' in self.unlocked_maneuvers:
            # sfx: quick_drop
            block = self.arm_holding_block
            x = int(self.arm_pos[0])
            lowest_y = self.GRID_HEIGHT - 1
            for y in range(int(self.arm_pos[1]), self.GRID_HEIGHT):
                if self.grid[y, x] != 0:
                    lowest_y = y - 1
                    break
            
            block['pos'] = [x, lowest_y]
            self._solidify_block(block)
            self.arm_holding_block = None
            self._check_for_line_clears()
            if not self.falling_block and not self.arm_holding_block:
                self._spawn_block()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
    def _update_game_state(self):
        # --- Block Falling ---
        if self.falling_block:
            self.falling_block['pos'][1] += self.fall_speed_current
            
            block_x = int(self.falling_block['pos'][0])
            block_y = int(self.falling_block['pos'][1])

            # --- Collision Check ---
            collided = False
            if block_y >= self.GRID_HEIGHT - 1:
                collided = True
                self.falling_block['pos'][1] = self.GRID_HEIGHT - 1
            elif block_y + 1 < self.GRID_HEIGHT and self.grid[block_y + 1, block_x] != 0:
                collided = True
                self.falling_block['pos'][1] = block_y

            if collided:
                if not self.falling_block.get('was_caught', False):
                    self.reward_this_step -= 0.1
                self._solidify_block(self.falling_block)
                self.falling_block = None
                self._check_for_line_clears()
        
        # --- Spawn new block if needed ---
        if not self.falling_block and not self.arm_holding_block and self.clear_line_timer == 0:
            self._spawn_block()
            
    def _spawn_block(self, is_initial=False):
        if is_initial:
            self.next_block = self._create_random_block()

        self.falling_block = self.next_block
        self.falling_block['pos'] = [float(self.np_random.integers(0, self.GRID_WIDTH)), -1.0]
        self.falling_block['was_caught'] = False
        
        self.next_block = self._create_random_block()

    def _create_random_block(self):
        block_type = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
        return {'type': block_type, 'color': self.BLOCK_COLORS[block_type-1]}

    def _solidify_block(self, block):
        # sfx: block_land
        x, y = int(block['pos'][0]), int(block['pos'][1])
        if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
            if self.grid[y, x] == 0:
                self.grid[y, x] = block['type']
                self._create_particles(self._grid_to_world([x, y]), block['color'], 20)
                if y < 1:
                    self.game_over = True
            else: # Trying to place on an existing block
                self.game_over = True
        else: # Block solidified out of bounds (e.g., above screen)
            self.game_over = True
            
    def _check_for_line_clears(self):
        full_rows = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] > 0):
                full_rows.append(r)
        
        if full_rows:
            # sfx: line_clear_charge
            self.lines_to_clear = full_rows
            self.clear_line_timer = 15 # frames for animation
            
            num_cleared = len(full_rows)
            if num_cleared == 1: self.reward_this_step += 1
            elif num_cleared == 2: self.reward_this_step += 2
            else: self.reward_this_step += 5
            self.score += (10 * num_cleared) * num_cleared
    
    def _execute_line_clear(self):
        # sfx: line_clear_boom
        for r in self.lines_to_clear:
            self._create_particles((self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE / 2, self.GRID_Y_OFFSET + r * self.CELL_SIZE), (255,255,255), 50, horizontal_line=True)
        
        new_grid = np.zeros_like(self.grid)
        new_row = self.GRID_HEIGHT - 1
        for r in range(self.GRID_HEIGHT - 1, -1, -1):
            if r not in self.lines_to_clear:
                new_grid[new_row, :] = self.grid[r, :]
                new_row -= 1
        self.grid = new_grid
        
        prev_level = self.lines_cleared // 5
        self.lines_cleared += len(self.lines_to_clear)
        new_level = self.lines_cleared // 5
        
        if new_level > prev_level:
            # sfx: level_up
            self.fall_speed_current = self.fall_speed_base + (new_level * 0.002)
        
        if self.lines_cleared >= 10 and 'quick_drop' not in self.unlocked_maneuvers:
            # sfx: maneuver_unlocked
            self.unlocked_maneuvers.add('quick_drop')

        self.lines_to_clear = []

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines_cleared": self.lines_cleared}
        
    def _grid_to_world(self, grid_pos):
        x = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        return int(x), int(y)

    def _render_text(self, text, pos, font, color, shadow_color=None, center=False, right=False):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            text_rect_shadow = text_surf_shadow.get_rect()
            if center: text_rect_shadow.center = (pos[0]+2, pos[1]+2)
            elif right: text_rect_shadow.topright = (pos[0]+2, pos[1]+2)
            else: text_rect_shadow.topleft = (pos[0]+2, pos[1]+2)
            self.screen.blit(text_surf_shadow, text_rect_shadow)

        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center: text_rect.center = pos
        elif right: text_rect.topright = pos
        else: text_rect.topleft = pos
        self.screen.blit(text_surf, text_rect)

    def _render_game(self):
        # --- Grid ---
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, y))

        # --- Solidified Blocks ---
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color = self.BLOCK_COLORS[self.grid[r,c]-1]
                    rect = pygame.Rect(self.GRID_X_OFFSET + c * self.CELL_SIZE + 1, self.GRID_Y_OFFSET + r * self.CELL_SIZE + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2)
                    pygame.draw.rect(self.screen, color, rect, border_radius=3)
        
        # --- Line Clear Flash ---
        if self.clear_line_timer > 0:
            flash_alpha = 150 * (self.clear_line_timer % 10) / 10
            flash_color = (255, 255, 255, flash_alpha)
            for r in self.lines_to_clear:
                flash_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET + r * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                s = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
                s.fill(flash_color)
                self.screen.blit(s, flash_rect.topleft)

        # --- Robotic Arm ---
        arm_base_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 5)
        arm_world_pos = self._grid_to_world(self.arm_pos)
        pygame.draw.line(self.screen, self.COLOR_ARM, arm_base_pos, arm_world_pos, 5)
        pygame.gfxdraw.filled_circle(self.screen, arm_world_pos[0], arm_world_pos[1], 12, (*self.COLOR_ARM_GLOW, 50))
        
        # Claw
        claw_open = self.arm_holding_block is None
        claw_angle = 20 if claw_open else -10
        claw_len = 15
        for i in [-1, 1]:
            angle = math.radians(90 + claw_angle * i)
            end_x = arm_world_pos[0] + math.cos(angle) * claw_len
            end_y = arm_world_pos[1] + math.sin(angle) * claw_len
            pygame.draw.line(self.screen, self.COLOR_ARM_GLOW, arm_world_pos, (int(end_x), int(end_y)), 4)

        # --- Falling/Held Blocks ---
        block_to_draw = self.falling_block or self.arm_holding_block
        if block_to_draw:
            pos = self.falling_block['pos'] if self.falling_block else self.arm_pos
            wx, wy = self._grid_to_world(pos)
            color = block_to_draw['color']
            
            glow_size = int(self.CELL_SIZE * 0.8)
            pygame.gfxdraw.filled_circle(self.screen, wx, wy, glow_size, (*color, 60))

            rect = pygame.Rect(wx - (self.CELL_SIZE-2)/2, wy - (self.CELL_SIZE-2)/2, self.CELL_SIZE-2, self.CELL_SIZE-2)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
        self._draw_particles()

    def _render_ui(self):
        # --- Score and Level ---
        self._render_text(f"SCORE: {self.score}", (20, 20), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        level = self.lines_cleared // 5
        self._render_text(f"LEVEL: {level}", (self.SCREEN_WIDTH - 20, 20), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, right=True)

        # --- Next Block Preview ---
        preview_box = pygame.Rect(self.SCREEN_WIDTH - 140, self.SCREEN_HEIGHT - 80, 120, 60)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box, border_radius=5)
        self._render_text("NEXT", preview_box.midtop + pygame.Vector2(0, -18), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)
        if self.next_block:
            color = self.next_block['color']
            rect = pygame.Rect(0, 0, self.CELL_SIZE-2, self.CELL_SIZE-2)
            rect.center = preview_box.center
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
        # --- Unlocked Maneuvers ---
        if 'quick_drop' in self.unlocked_maneuvers:
            self._render_text("QUICK DROP [SHIFT]", (20, self.SCREEN_HEIGHT - 30), self.font_small, (100, 255, 100), self.COLOR_TEXT_SHADOW)

        # --- Game Over ---
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0,0))
            self._render_text("GAME OVER", (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), self.font_gameover, (255, 50, 50), self.COLOR_TEXT_SHADOW, center=True)

    def _create_particles(self, pos, color, count, horizontal_line=False):
        for _ in range(count):
            if horizontal_line:
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-3, -1)]
                start_pos = [pos[0] + self.np_random.uniform(-self.GRID_WIDTH*self.CELL_SIZE/2, self.GRID_WIDTH*self.CELL_SIZE/2), pos[1]]
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                start_pos = list(pos)
            self.particles.append({
                'pos': start_pos,
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(3, 7)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['lifespan'] -= 1
            p['size'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['size'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'])
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)


if __name__ == "__main__":
    # The following code is for manual testing and visualization.
    # It is not part of the Gymnasium environment itself.
    
    # To run this, you'll need to unset the dummy video driver
    # and use a real one.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Robo-Stacker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while not done:
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        # This is a manual mapping from keyboard to action space
        # For a real agent, you would use a policy to select an action
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
        
        # Blit the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()
    print(f"Game Over! Final Score: {info['score']}, Lines Cleared: {info['lines_cleared']}")