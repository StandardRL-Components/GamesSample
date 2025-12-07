
# Generated: 2025-08-27T19:23:37.382681
# Source Brief: brief_02138.md
# Brief Index: 2138

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the block. Hold space to drop it quickly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match 3 or more falling colored blocks in a grid to score points. Reach 100 points to win, but don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 223, 0)
    
    # Block Colors (index 0 is empty)
    BLOCK_COLORS = [
        (0, 0, 0),  # Empty
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    # Game parameters
    WIN_SCORE = 100
    MAX_STEPS = 10000
    INITIAL_FALL_SPEED = 0.03  # cells per step
    FALL_SPEED_INCREASE = 0.0005 # speed increase per step

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Grid position
        self.grid_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.grid_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        # Etc...        
        self.grid = None
        self.piece = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0
        self.particles = []
        self.animation_grid = None
        self.space_was_held = False
        self.rng = None
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Initialize all game state
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.animation_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.particles = []
        self.space_was_held = False
        
        self._spawn_new_piece()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        self.steps += 1
        reward = 0
        
        # --- Handle Input ---
        # Movement is discrete, happens once per step
        if movement == 3: # Left
            self._move_piece(-1)
        elif movement == 4: # Right
            self._move_piece(1)
            
        # Hard drop on space press (rising edge)
        if space_held and not self.space_was_held:
            # Sound: Player_HardDrop
            while not self._check_collision(0, 1):
                self.piece['y'] += 1
            reward -= 0.05 # Small penalty for haste
        self.space_was_held = space_held

        # --- Game Logic ---
        self.fall_speed = self.INITIAL_FALL_SPEED + self.steps * self.FALL_SPEED_INCREASE
        self.piece['y'] += self.fall_speed

        # Check for landing
        if self._check_collision(0, 1):
            lock_y = math.floor(self.piece['y'])
            if lock_y < 0: # Game over condition
                self.game_over = True
                reward -= 50
            else:
                # Lock piece in place
                self.grid[lock_y, self.piece['x']] = self.piece['color']
                # Sound: Block_Lock
                
                # Check for matches and chains
                chain_reward, blocks_cleared = self._handle_matches()
                
                if blocks_cleared == 0:
                    reward -= 0.01 # Penalty for placing a block that doesn't match
                else:
                    reward += chain_reward
                
                self._spawn_new_piece()
        
        self._update_animations()
        self._update_particles()
        
        # --- Termination ---
        terminated = self.game_over or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        if self.score >= self.WIN_SCORE and not self.game_over:
            reward += 100 # Win bonus
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _spawn_new_piece(self):
        self.piece = {
            'x': self.GRID_WIDTH // 2,
            'y': -1.0,
            'color': self.rng.integers(1, len(self.BLOCK_COLORS))
        }
        if self._check_collision(0, 0): # Game over if spawn is blocked
            self.game_over = True

    def _move_piece(self, dx):
        if not self._check_collision(dx, 0):
            self.piece['x'] += dx
            # Sound: Player_Move

    def _check_collision(self, dx, dy):
        if self.piece is None: return True
        
        new_x = self.piece['x'] + dx
        new_y = math.floor(self.piece['y'] + dy)

        if not (0 <= new_x < self.GRID_WIDTH):
            return True
        if new_y >= self.GRID_HEIGHT:
            return True
        if new_y >= 0 and self.grid[new_y, new_x] > 0:
            return True
            
        return False

    def _handle_matches(self):
        total_reward = 0
        total_blocks_cleared = 0
        chain_multiplier = 1.0

        while True:
            matches = self._find_matches()
            if not matches:
                break

            # Sound: Match_Clear
            num_cleared = len(matches)
            total_blocks_cleared += num_cleared

            # Calculate reward
            base_reward = 0
            if num_cleared == 3: base_reward = 1
            elif num_cleared == 4: base_reward = 2
            elif num_cleared >= 5: base_reward = 5
            
            reward = (base_reward + num_cleared * 0.1) * chain_multiplier
            total_reward += reward
            self.score += round(reward) # Score is integer part of reward

            # Animate and clear blocks
            for r, c in matches:
                self.animation_grid[r, c] = 15 # Animation duration
                self._create_particles(c, r, self.grid[r, c])
                self.grid[r, c] = 0
            
            self._apply_gravity()
            chain_multiplier += 0.5 # Increase multiplier for next chain
            # Sound: Chain_Reaction

        return total_reward, total_blocks_cleared

    def _find_matches(self):
        to_clear = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color = self.grid[r, c]
                if color == 0: continue

                # Horizontal match
                h_match = [(r, c)]
                for i in range(1, self.GRID_WIDTH):
                    if c + i < self.GRID_WIDTH and self.grid[r, c+i] == color:
                        h_match.append((r, c+i))
                    else: break
                if len(h_match) >= 3:
                    to_clear.update(h_match)

                # Vertical match
                v_match = [(r, c)]
                for i in range(1, self.GRID_HEIGHT):
                    if r + i < self.GRID_HEIGHT and self.grid[r+i, c] == color:
                        v_match.append((r+i, c))
                    else: break
                if len(v_match) >= 3:
                    to_clear.update(v_match)
        return to_clear
        
    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            write_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] > 0:
                    self.grid[write_row, c], self.grid[r, c] = self.grid[r, c], self.grid[write_row, c]
                    write_row -= 1

    def _update_animations(self):
        self.animation_grid = np.maximum(0, self.animation_grid - 1)

    def _create_particles(self, c, r, color_idx):
        px = self.grid_x + c * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.grid_y + r * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.BLOCK_COLORS[color_idx]
        for _ in range(10): # 10 particles per cleared block
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.rng.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_x, self.grid_y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.grid_y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (self.grid_x, y), (self.grid_x + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.grid_x + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (x, self.grid_y), (x, self.grid_y + self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw settled blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    self._draw_block(c, r, color_idx, self.animation_grid[r,c])
        
        # Draw falling piece
        if self.piece and not self.game_over:
            # Ghost piece
            ghost_y = self.piece['y']
            while not self._check_collision(0, ghost_y - self.piece['y'] + 1):
                ghost_y += 1
            self._draw_block(self.piece['x'], math.floor(ghost_y), self.piece['color'], 0, is_ghost=True)

            # Actual piece
            self._draw_block(self.piece['x'], self.piece['y'], self.piece['color'], 0)

        # Draw particles
        self._render_particles()

    def _draw_block(self, c, r, color_idx, anim_timer, is_ghost=False):
        x = self.grid_x + c * self.CELL_SIZE
        y = self.grid_y + r * self.CELL_SIZE
        
        color = self.BLOCK_COLORS[color_idx]
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect.inflate(-self.CELL_SIZE*0.8, -self.CELL_SIZE*0.8), 2, border_radius=3)
            return

        if anim_timer > 0:
            # Flashing animation
            p = anim_timer / 15.0
            flash_color = (
                min(255, color[0] + (255 - color[0]) * p),
                min(255, color[1] + (255 - color[1]) * p),
                min(255, color[2] + (255 - color[2]) * p),
            )
            pygame.draw.rect(self.screen, flash_color, rect, border_radius=4)
        else:
            # Standard block with highlight and shadow
            shadow_color = tuple(max(0, val-40) for val in color)
            highlight_color = tuple(min(255, val+40) for val in color)
            pygame.draw.rect(self.screen, shadow_color, rect, border_radius=4)
            pygame.draw.rect(self.screen, highlight_color, rect.inflate(-2,-2), border_radius=3)
            pygame.draw.rect(self.screen, color, rect.inflate(-4,-4), border_radius=2)
            
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = p['color'] + (alpha,)
            size = max(1, int(5 * (p['life'] / 30.0)))
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 20))
        
        # Steps
        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (20, 50))
        
        # Game Over / Win message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                msg_color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                msg_color = (255, 100, 100)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_title.render(msg, True, msg_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Requires a display. Will not work in a purely headless environment.
    import os
    # If you are in a headless environment, you might need to set this
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    # env.validate_implementation() # Uncomment to run validation
    
    obs, info = env.reset()
    done = False
    
    # --- Human Playback Setup ---
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("ColorFall")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            # --- Action Mapping for Human ---
            keys = pygame.key.get_pressed()
            move_action = 0 # No-op
            if keys[pygame.K_LEFT]: move_action = 3
            elif keys[pygame.K_RIGHT]: move_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 0 # Not used

            action = [move_action, space_action, shift_action]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
            
            if not done:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                if done:
                    print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")

            # --- Rendering ---
            # The observation is already a rendered frame, so we just need to display it.
            # Pygame uses (width, height), but our obs is (height, width, 3).
            # We need to transpose it back for display.
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Run at 30 FPS for human play

    except pygame.error as e:
        print(f"\nPygame display error: {e}. This is expected in a headless environment.")
        print("The environment is functional for RL training but cannot be played directly.")
        print("To play, run this script in an environment with a display server (e.g., your local machine).\n")

    env.close()