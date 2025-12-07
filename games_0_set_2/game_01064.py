
# Generated: 2025-08-27T15:44:57.218944
# Source Brief: brief_01064.md
# Brief Index: 1064

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant, fast-paced puzzle game. Rotate and drop blocks to clear lines, "
        "aiming to complete 3 progressively faster stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 10, 20
    BLOCK_SIZE = 18
    PLAYFIELD_PX_WIDTH = PLAYFIELD_WIDTH * BLOCK_SIZE
    PLAYFIELD_PX_HEIGHT = PLAYFIELD_HEIGHT * BLOCK_SIZE
    PLAYFIELD_X = (SCREEN_WIDTH - PLAYFIELD_PX_WIDTH) // 2
    PLAYFIELD_Y = (SCREEN_HEIGHT - PLAYFIELD_PX_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 30, 45, 180) # with alpha
    COLOR_GHOST = (255, 255, 255, 60) # with alpha
    
    BLOCK_COLORS = [
        (230, 60, 60),   # Red
        (60, 230, 60),   # Green
        (60, 60, 230),   # Blue
        (230, 230, 60),  # Yellow
        (230, 60, 230),  # Magenta
        (60, 230, 230),  # Cyan
        (240, 140, 40)   # Orange
    ]

    # --- Tetromino Shapes ---
    # Stored as lists of rotations, where each rotation is a list of (row, col) offsets
    SHAPES = [
        [[(0, 0), (-1, 0), (1, 0), (2, 0)], [(0, 0), (0, -1), (0, 1), (0, 2)]],  # I
        [[(0, 0), (0, 1), (1, 0), (1, 1)]],  # O
        [[(0, 0), (-1, 0), (1, 0), (0, -1)], [(0, 0), (0, -1), (0, 1), (1, 0)],
         [(0, 0), (-1, 0), (1, 0), (0, 1)], [(0, 0), (0, -1), (0, 1), (-1, 0)]], # T
        [[(0, 0), (-1, 0), (1, 0), (1, -1)], [(0, 0), (0, -1), (0, 1), (1, 1)],
         [(0, 0), (-1, 0), (1, 0), (-1, 1)], [(0, 0), (0, -1), (0, 1), (-1, -1)]], # J
        [[(0, 0), (-1, 0), (1, 0), (-1, -1)], [(0, 0), (0, -1), (0, 1), (1, -1)],
         [(0, 0), (-1, 0), (1, 0), (1, 1)], [(0, 0), (0, -1), (0, 1), (-1, 1)]], # L
        [[(0, 0), (-1, 0), (0, 1), (1, 1)], [(0, 0), (0, -1), (1, 0), (1, 1)]],  # S
        [[(0, 0), (1, 0), (0, 1), (-1, 1)], [(0, 0), (0, 1), (1, 0), (1, -1)]]   # Z
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
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 20)

        self.np_random = None
        
        # State variables are initialized in reset()
        self.reset()
        
        # This must be called at the end of __init__
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.playfield = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        
        self.steps = 0
        self.score = 0
        self.total_lines_cleared = 0
        self.stage_lines_cleared = 0
        self.stage = 1
        self.game_over = False
        self.victory = False
        
        self.drop_timer = 0.0
        self.stage_speeds = {1: 1.0, 2: 1.5, 3: 2.0} # cells per second

        self.bag = list(range(len(self.SHAPES)))
        random.shuffle(self.bag)
        
        self.current_block = self._new_block()
        self.next_block = self._new_block()
        
        self.prev_space_held = False
        self.prev_up_held = False
        self.move_cooldown = 0
        
        self.particles = []
        self.line_clear_animation = None # (y_indices, timer)

        return self._get_observation(), self._get_info()

    def _new_block(self):
        if not self.bag:
            self.bag = list(range(len(self.SHAPES)))
            random.shuffle(self.bag)
        shape_index = self.bag.pop()
        
        return {
            "shape_index": shape_index,
            "rotation": 0,
            "pos": [0, self.PLAYFIELD_WIDTH // 2 - 1],
            "color_index": shape_index % len(self.BLOCK_COLORS)
        }

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if self.game_over or self.victory:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self._update_animations()

        if self.line_clear_animation:
            return self._get_observation(), 0, False, False, self._get_info()

        # --- Action Handling ---
        hard_dropped = False
        if space_held and not self.prev_space_held: # Hard drop on key press
            # Sound: Hard drop
            while not self._check_collision(self.current_block['rotation'], self.current_block['pos']):
                self.current_block['pos'][0] += 1
            self.current_block['pos'][0] -= 1
            hard_dropped = True
        self.prev_space_held = space_held

        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        if not hard_dropped:
            # Rotation
            up_held = movement == 1
            if up_held and not self.prev_up_held:
                self._rotate_block()
            self.prev_up_held = up_held

            # Horizontal Movement
            if self.move_cooldown == 0:
                if movement == 3: # Left
                    new_pos = [self.current_block['pos'][0], self.current_block['pos'][1] - 1]
                    if not self._check_collision(self.current_block['rotation'], new_pos):
                        self.current_block['pos'] = new_pos
                        reward -= 0.02
                        self.move_cooldown = 3 # 3 frames cooldown
                elif movement == 4: # Right
                    new_pos = [self.current_block['pos'][0], self.current_block['pos'][1] + 1]
                    if not self._check_collision(self.current_block['rotation'], new_pos):
                        self.current_block['pos'] = new_pos
                        reward -= 0.02
                        self.move_cooldown = 3 # 3 frames cooldown
            
            # Soft Drop
            is_soft_dropping = movement == 2
            drop_speed = self.stage_speeds[self.stage] * (5.0 if is_soft_dropping else 1.0)
            if is_soft_dropping:
                reward += 0.1

            # Automatic Drop
            self.drop_timer += drop_speed / 30.0 # Assuming 30 FPS
            if self.drop_timer >= 1.0:
                self.drop_timer = 0
                new_pos = [self.current_block['pos'][0] + 1, self.current_block['pos'][1]]
                if not self._check_collision(self.current_block['rotation'], new_pos):
                    self.current_block['pos'] = new_pos
                else:
                    hard_dropped = True # Lock if it collides after auto-drop

        # --- Lock Piece Logic ---
        if hard_dropped:
            self._lock_block()
            cleared_count = self._check_and_clear_lines()

            if cleared_count > 0:
                # Sound: Line clear
                line_rewards = {1: 1, 2: 2, 3: 4, 4: 8}
                reward += line_rewards.get(cleared_count, 0)
                self.score += [0, 100, 300, 500, 800][cleared_count] * self.stage
                self.total_lines_cleared += cleared_count
                self.stage_lines_cleared += cleared_count
            else:
                # Penalty for placing a block without clearing a line
                reward -= 0.2

            # Stage Progression
            if self.stage_lines_cleared >= 10:
                self.stage += 1
                self.stage_lines_cleared = 0
                reward += 10
                if self.stage > 3:
                    self.victory = True
                    reward += 100
                else:
                    # Sound: Stage up
                    pass

            if not self.victory:
                self.current_block = self.next_block
                self.next_block = self._new_block()
                if self._check_collision(self.current_block['rotation'], self.current_block['pos']):
                    self.game_over = True
                    reward -= 100
                    # Sound: Game over

        self.steps += 1
        terminated = self.game_over or self.victory or self.steps >= 5000
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_shape_coords(self, shape_index, rotation, pos):
        shape_def = self.SHAPES[shape_index]
        rotation_def = shape_def[rotation % len(shape_def)]
        return [(pos[0] + r, pos[1] + c) for r, c in rotation_def]
    
    def _check_collision(self, rotation, pos):
        coords = self._get_shape_coords(self.current_block['shape_index'], rotation, pos)
        for r, c in coords:
            if not (0 <= c < self.PLAYFIELD_WIDTH and 0 <= r < self.PLAYFIELD_HEIGHT):
                return True # Out of bounds
            if self.playfield[r, c] != 0:
                return True # Collides with existing block
        return False

    def _rotate_block(self):
        # Sound: Rotate
        new_rotation = (self.current_block['rotation'] + 1)
        
        # Wall kick logic (basic)
        offsets = [(0, 0), (0, -1), (0, 1), (0, -2), (0, 2)]
        for r_off, c_off in offsets:
            new_pos = [self.current_block['pos'][0] + r_off, self.current_block['pos'][1] + c_off]
            if not self._check_collision(new_rotation, new_pos):
                self.current_block['rotation'] = new_rotation % len(self.SHAPES[self.current_block['shape_index']])
                self.current_block['pos'] = new_pos
                return

    def _lock_block(self):
        # Sound: Block lock
        coords = self._get_shape_coords(self.current_block['shape_index'], self.current_block['rotation'], self.current_block['pos'])
        for r, c in coords:
            if 0 <= r < self.PLAYFIELD_HEIGHT and 0 <= c < self.PLAYFIELD_WIDTH:
                self.playfield[r, c] = self.current_block['color_index'] + 1

    def _check_and_clear_lines(self):
        lines_to_clear = [r for r in range(self.PLAYFIELD_HEIGHT) if np.all(self.playfield[r, :] != 0)]
        if not lines_to_clear:
            return 0
        
        self.line_clear_animation = (lines_to_clear, 10) # 10 frames animation

        for r in lines_to_clear:
            self._create_particles(r)
            
        new_playfield = np.zeros_like(self.playfield)
        new_row = self.PLAYFIELD_HEIGHT - 1
        for r in range(self.PLAYFIELD_HEIGHT - 1, -1, -1):
            if r not in lines_to_clear:
                new_playfield[new_row, :] = self.playfield[r, :]
                new_row -= 1
        self.playfield = new_playfield
        
        return len(lines_to_clear)

    def _get_ghost_position(self):
        ghost_pos = list(self.current_block['pos'])
        while not self._check_collision(self.current_block['rotation'], ghost_pos):
            ghost_pos[0] += 1
        ghost_pos[0] -= 1
        return ghost_pos

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines": self.total_lines_cleared,
            "stage": self.stage
        }

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _update_animations(self):
        # Line clear animation
        if self.line_clear_animation:
            indices, timer = self.line_clear_animation
            timer -= 1
            if timer <= 0:
                self.line_clear_animation = None
            else:
                self.line_clear_animation = (indices, timer)

        # Particle physics
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, y_row):
        y_px = self.PLAYFIELD_Y + y_row * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        for x in range(self.PLAYFIELD_WIDTH):
            x_px = self.PLAYFIELD_X + x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
            color = self.BLOCK_COLORS[self.playfield[y_row, x] - 1]
            for _ in range(3): # 3 particles per block
                particle = {
                    'pos': [x_px, y_px],
                    'vel': [random.uniform(-2, 2), random.uniform(-3, -1)],
                    'life': random.randint(20, 40),
                    'color': color
                }
                self.particles.append(particle)

    def _draw_block(self, surface, pos, color, size, is_ghost=False):
        rect = pygame.Rect(pos[0], pos[1], size, size)
        if is_ghost:
            pygame.gfxdraw.box(surface, rect, color)
        else:
            light_color = tuple(min(255, c + 60) for c in color)
            dark_color = tuple(max(0, c - 60) for c in color)
            
            # Main block face
            pygame.draw.rect(surface, color, rect)
            # 3D effect
            pygame.draw.line(surface, light_color, rect.topleft, rect.topright, 2)
            pygame.draw.line(surface, light_color, rect.topleft, rect.bottomleft, 2)
            pygame.draw.line(surface, dark_color, rect.bottomright, rect.topright, 2)
            pygame.draw.line(surface, dark_color, rect.bottomright, rect.bottomleft, 2)

    def _render_all(self):
        # Background
        bg_color_shift = (math.sin(self.steps * 0.01) + 1) / 2 * 15
        self.screen.fill(tuple(max(0, c + bg_color_shift) for c in self.COLOR_BG))

        # Playfield Grid
        for x in range(self.PLAYFIELD_WIDTH + 1):
            px = self.PLAYFIELD_X + x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.PLAYFIELD_Y), (px, self.PLAYFIELD_Y + self.PLAYFIELD_PX_HEIGHT))
        for y in range(self.PLAYFIELD_HEIGHT + 1):
            py = self.PLAYFIELD_Y + y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X, py), (self.PLAYFIELD_X + self.PLAYFIELD_PX_WIDTH, py))
        
        # Locked Blocks
        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                if self.playfield[r, c] != 0:
                    color_index = int(self.playfield[r, c] - 1)
                    color = self.BLOCK_COLORS[color_index]
                    px = self.PLAYFIELD_X + c * self.BLOCK_SIZE
                    py = self.PLAYFIELD_Y + r * self.BLOCK_SIZE
                    self._draw_block(self.screen, (px, py), color, self.BLOCK_SIZE)

        # Line clear animation flash
        if self.line_clear_animation:
            indices, timer = self.line_clear_animation
            alpha = int(255 * (timer / 10.0))
            flash_surface = pygame.Surface((self.PLAYFIELD_PX_WIDTH, self.BLOCK_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            for r in indices:
                self.screen.blit(flash_surface, (self.PLAYFIELD_X, self.PLAYFIELD_Y + r * self.BLOCK_SIZE))

        # Ghost and Current Block
        if not self.game_over and not self.victory:
            # Ghost
            ghost_pos = self._get_ghost_position()
            ghost_coords = self._get_shape_coords(self.current_block['shape_index'], self.current_block['rotation'], ghost_pos)
            for r, c in ghost_coords:
                px = self.PLAYFIELD_X + c * self.BLOCK_SIZE
                py = self.PLAYFIELD_Y + r * self.BLOCK_SIZE
                self._draw_block(self.screen, (px, py), self.COLOR_GHOST, self.BLOCK_SIZE, is_ghost=True)

            # Current
            current_coords = self._get_shape_coords(self.current_block['shape_index'], self.current_block['rotation'], self.current_block['pos'])
            color = self.BLOCK_COLORS[self.current_block['color_index']]
            for r, c in current_coords:
                px = self.PLAYFIELD_X + c * self.BLOCK_SIZE
                py = self.PLAYFIELD_Y + r * self.BLOCK_SIZE
                self._draw_block(self.screen, (px, py), color, self.BLOCK_SIZE)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, p['life'] * 10))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, int(p['life'] / 10.0))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0, 0, size*2, size*2))
            self.screen.blit(temp_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

        # UI
        self._render_ui()

    def _render_ui(self):
        # Next Piece
        next_box_rect = pygame.Rect(self.PLAYFIELD_X + self.PLAYFIELD_PX_WIDTH + 20, self.PLAYFIELD_Y, 120, 120)
        s = pygame.Surface(next_box_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, next_box_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box_rect, 2)
        
        next_text = self.font_main.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (next_box_rect.centerx - next_text.get_width() // 2, next_box_rect.y + 10))
        
        next_coords = self._get_shape_coords(self.next_block['shape_index'], 0, (2, 2))
        color = self.BLOCK_COLORS[self.next_block['color_index']]
        for r, c in next_coords:
            px = next_box_rect.x + c * (self.BLOCK_SIZE * 0.75)
            py = next_box_rect.y + 40 + r * (self.BLOCK_SIZE * 0.75)
            self._draw_block(self.screen, (px, py), color, int(self.BLOCK_SIZE * 0.75))

        # Score, Lines, Stage
        info_box_rect = pygame.Rect(self.PLAYFIELD_X - 140, self.PLAYFIELD_Y, 120, 160)
        s = pygame.Surface(info_box_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, info_box_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_GRID, info_box_rect, 2)
        
        score_text = self.font_main.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_small.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        lines_text = self.font_main.render("LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_small.render(f"{self.total_lines_cleared}", True, self.COLOR_UI_TEXT)
        stage_text = self.font_main.render("STAGE", True, self.COLOR_UI_TEXT)
        stage_val = self.font_small.render(f"{self.stage}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_text, (info_box_rect.centerx - score_text.get_width() // 2, info_box_rect.y + 10))
        self.screen.blit(score_val, (info_box_rect.centerx - score_val.get_width() // 2, info_box_rect.y + 35))
        self.screen.blit(lines_text, (info_box_rect.centerx - lines_text.get_width() // 2, info_box_rect.y + 60))
        self.screen.blit(lines_val, (info_box_rect.centerx - lines_val.get_width() // 2, info_box_rect.y + 85))
        self.screen.blit(stage_text, (info_box_rect.centerx - stage_text.get_width() // 2, info_box_rect.y + 110))
        self.screen.blit(stage_val, (info_box_rect.centerx - stage_val.get_width() // 2, info_box_rect.y + 135))

        # Game Over / Victory Message
        if self.game_over:
            msg = self.font_title.render("GAME OVER", True, (255, 50, 50))
            self.screen.blit(msg, (self.SCREEN_WIDTH // 2 - msg.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg.get_height() // 2))
        elif self.victory:
            msg = self.font_title.render("VICTORY!", True, (50, 255, 50))
            self.screen.blit(msg, (self.SCREEN_WIDTH // 2 - msg.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg.get_height() // 2))

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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gymnasium Tetris")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input to Action ---
        movement = 0 # no-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Optional: auto-reset
            # obs, info = env.reset()
            # total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()