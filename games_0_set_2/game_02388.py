
# Generated: 2025-08-27T20:12:16.869439
# Source Brief: brief_02388.md
# Brief Index: 2388

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character (white circle) one tile at a time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated grid filled with invisible traps to reach the exit (glowing square) "
        "before the timer runs out. Each stage gets harder. Good luck."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Visual and Gameplay Constants
        self.GRID_SIZE = 10
        self.MAX_STAGES = 3
        self.MAX_EPISODE_STEPS = 180
        self.STAGE_TIME_LIMIT = 60
        self.INITIAL_TRAPS = 10
        self.TRAPS_PER_STAGE = 2

        self._init_colors_and_fonts()
        
        # Initialize state variables
        self.reset()
        
        # Run self-check
        self.validate_implementation()
    
    def _init_colors_and_fonts(self):
        """Initialize all color and font constants for rendering."""
        self.COLOR_BG = (15, 18, 26)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_EXIT = (220, 220, 255)
        self.COLOR_TRAP_REVEAL = (255, 50, 50)
        self.COLOR_TEXT = (200, 200, 220)
        self.COLOR_TEXT_ACCENT = (255, 255, 100)
        self.COLOR_GAMEOVER = (255, 80, 80)
        self.COLOR_WIN = (80, 255, 80)

        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_LARGE = pygame.font.Font(None, 72)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        
        self.revealed_traps = {} # {pos: ttl}
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Sets up a new stage with player, exit, and traps."""
        self.stage_timer = self.STAGE_TIME_LIMIT
        num_traps = self.INITIAL_TRAPS + (self.stage - 1) * self.TRAPS_PER_STAGE
        
        # Generate all possible grid positions
        all_pos = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_pos)
        
        # Assign unique positions
        self.player_pos = all_pos.pop()
        self.exit_pos = all_pos.pop()
        self.trap_locations = set(all_pos[:num_traps])
        
        # Clear any lingering visual effects from the previous stage
        self.revealed_traps.clear()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        self.steps += 1
        self.stage_timer -= 1
        
        # Update revealed trap effects
        self._update_effects()

        # Handle movement
        prev_pos = self.player_pos
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_x = max(0, min(self.GRID_SIZE - 1, self.player_pos[0] + dx))
            new_y = max(0, min(self.GRID_SIZE - 1, self.player_pos[1] + dy))
            self.player_pos = (new_x, new_y)

        # Process game events based on the new position
        if self.player_pos in self.trap_locations:
            # Player stepped on a trap
            reward = -100
            self.game_over = True
            self.revealed_traps[self.player_pos] = 20 # Show the trap on game over
            # sfx: player_death_squish.wav
        elif self.player_pos == self.exit_pos:
            # Player reached the exit
            if self.stage == self.MAX_STAGES:
                reward = 50 # Final victory
                self.game_over = True
                self.win = True
                # sfx: victory_fanfare.wav
            else:
                reward = 5 # Stage complete
                self.stage += 1
                self._setup_stage()
                # sfx: stage_clear.wav
        elif self.player_pos != prev_pos:
            # Player moved to a safe tile
            reward = 0.1
            # sfx: player_step.wav
            
            # Check adjacent tiles for traps and reveal them
            for dx_check in [-1, 0, 1]:
                for dy_check in [-1, 0, 1]:
                    if dx_check == 0 and dy_check == 0:
                        continue
                    check_pos = (self.player_pos[0] + dx_check, self.player_pos[1] + dy_check)
                    if check_pos in self.trap_locations:
                        reward -= 0.2
                        if check_pos not in self.revealed_traps:
                            self.revealed_traps[check_pos] = 15 # Reveal for 15 steps
                            # sfx: trap_reveal_hiss.wav
        
        # Check for other termination conditions
        terminated = self.game_over
        if self.stage_timer <= 0:
            reward = -100
            terminated = True
            self.game_over = True
            # sfx: timer_buzz_death.wav
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_effects(self):
        """Update TTL for visual effects like revealed traps."""
        next_revealed = {}
        for pos, ttl in self.revealed_traps.items():
            if ttl > 1:
                next_revealed[pos] = ttl - 1
        self.revealed_traps = next_revealed

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
            "stage": self.stage,
            "time_left": self.stage_timer,
        }

    def _render_game(self):
        """Renders the main game grid and entities."""
        grid_pixel_size = 360
        cell_size = grid_pixel_size // self.GRID_SIZE
        offset_x = (self.screen.get_width() - grid_pixel_size) // 2
        offset_y = (self.screen.get_height() - grid_pixel_size) // 2 + 20

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            start_x, end_x = offset_x + i * cell_size, offset_x + i * cell_size
            start_y, end_y = offset_y, offset_y + grid_pixel_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))
            
            start_x, end_x = offset_x, offset_x + grid_pixel_size
            start_y, end_y = offset_y + i * cell_size, offset_y + i * cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))

        # Helper to convert grid coords to pixel center
        def grid_to_pixel(pos):
            return (
                int(offset_x + (pos[0] + 0.5) * cell_size),
                int(offset_y + (pos[1] + 0.5) * cell_size)
            )

        # Draw revealed traps (fading red circles)
        for pos, ttl in self.revealed_traps.items():
            center_px = grid_to_pixel(pos)
            radius = int(cell_size * 0.3)
            alpha = int(100 * (ttl / 15.0)) # Fade out effect
            pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius, (*self.COLOR_TRAP_REVEAL, alpha))
            pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius, (*self.COLOR_TRAP_REVEAL, alpha))

        # Draw exit (glowing square)
        exit_px = grid_to_pixel(self.exit_pos)
        exit_rect = pygame.Rect(0, 0, cell_size * 0.7, cell_size * 0.7)
        exit_rect.center = exit_px
        self._draw_glowing_rect(self.screen, self.COLOR_EXIT, exit_rect, 20)

        # Draw player (glowing circle)
        player_px = grid_to_pixel(self.player_pos)
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, player_px, int(cell_size * 0.35), 15)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_strength):
        """Draws a circle with a soft glow effect."""
        for i in range(glow_strength, 0, -1):
            alpha = int(100 * (1 - (i / glow_strength)))
            pygame.gfxdraw.aacircle(surface, center[0], center[1], radius + i, (*color, alpha))
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        
    def _draw_glowing_rect(self, surface, color, rect, glow_strength):
        """Draws a rectangle with a soft glow effect."""
        for i in range(glow_strength, 0, -1):
            alpha = int(80 * (1 - (i / glow_strength)))
            glow_rect = rect.inflate(i*2, i*2)
            pygame.draw.rect(surface, (*color, alpha), glow_rect, border_radius=5)
        pygame.draw.rect(surface, color, rect, border_radius=3)


    def _render_ui(self):
        """Renders the UI elements like score, timer, and game over text."""
        # Score
        self._render_text(f"SCORE: {self.score:.1f}", (10, 10), self.FONT_UI, self.COLOR_TEXT)
        # Stage
        self._render_text(f"STAGE: {self.stage}/{self.MAX_STAGES}", (self.screen.get_width() / 2, 10), self.FONT_UI, self.COLOR_TEXT, center_x=True)
        # Timer
        time_color = self.COLOR_TEXT_ACCENT if self.stage_timer > 10 else self.COLOR_GAMEOVER
        self._render_text(f"TIME: {self.stage_timer}", (self.screen.get_width() - 10, 10), self.FONT_UI, time_color, center_x="right")
        
        if self.game_over:
            if self.win:
                self._render_text("VICTORY!", (self.screen.get_width() / 2, self.screen.get_height() / 2), self.FONT_LARGE, self.COLOR_WIN, center_x=True, center_y=True)
            else:
                self._render_text("GAME OVER", (self.screen.get_width() / 2, self.screen.get_height() / 2), self.FONT_LARGE, self.COLOR_GAMEOVER, center_x=True, center_y=True)

    def _render_text(self, text, position, font, color, center_x=False, center_y=False):
        """Helper to render text with various alignment options."""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        
        if center_x == True:
            text_rect.centerx = position[0]
        elif center_x == "right":
            text_rect.right = position[0]
        else:
            text_rect.left = position[0]
            
        if center_y:
            text_rect.centery = position[1]
        else:
            text_rect.top = position[1]
            
        self.screen.blit(text_surface, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Grid Nightmare")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                
                # Only register one action per frame
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        if not terminated:
            # Since auto_advance is False, we only step when an action is taken
            # or if we wanted to just wait (action = [0,0,0])
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 fps
        
    pygame.quit()