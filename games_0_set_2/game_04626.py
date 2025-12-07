
# Generated: 2025-08-28T02:57:59.330115
# Source Brief: brief_04626.md
# Brief Index: 4626

        
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
        "Controls: Press space to jump over obstacles. Other keys have no effect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling racer. Jump over obstacles to complete 3 laps in under 60 seconds."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (40, 20, 80)
    COLOR_BG_BOTTOM = (80, 40, 120)
    COLOR_TRACK = (180, 180, 190)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_OUTLINE = (255, 200, 200)
    COLOR_OBSTACLE = (20, 20, 40)
    COLOR_OBSTACLE_OUTLINE = (60, 60, 80)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)
    
    # Player
    PLAYER_WIDTH = 40
    PLAYER_HEIGHT = 20
    PLAYER_X_POS = 100
    PLAYER_JUMP_STRENGTH = -12
    PLAYER_GRAVITY = 0.6
    
    # Track
    TRACK_Y = 350
    TRACK_THICKNESS = 10
    TRACK_LENGTH = 4000
    
    # Game
    FPS = 30
    TIME_LIMIT = 60.0
    MAX_LAPS = 3
    MAX_STEPS = int(TIME_LIMIT * FPS) + 10 # Safety buffer

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.lap = 1
        self.track_progress = 0
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = True
        self.player_squash = 0
        self.obstacles = []
        self.particles = []
        self.cleared_obstacles = set()
        self.win_condition = False
        
        # Initialize state variables
        self.reset()

        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.time_elapsed = 0.0
        self.lap = 1
        self.track_progress = 0
        
        # Player state
        self.player_y = self.TRACK_Y - self.PLAYER_HEIGHT
        self.player_vy = 0
        self.on_ground = True
        self.player_squash = 0
        
        # World state
        self.obstacles = []
        self.particles = []
        self.cleared_obstacles = set()
        self._generate_track()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        # movement = action[0]  # Unused
        space_pressed = action[1] == 1
        # shift_held = action[2] == 1  # Unused

        reward = 0
        terminated = False

        if not self.game_over:
            # --- Update game logic ---
            self.steps += 1
            self.time_elapsed += 1 / self.FPS
            reward += 0.1  # Survival reward

            # Handle player input
            if space_pressed and self.on_ground:
                self.player_vy = self.PLAYER_JUMP_STRENGTH
                self.on_ground = False
                self.player_squash = 5 # Stretch effect
                # Sound: Jump
                # Create jump particles
                for _ in range(10):
                    self._create_particle(
                        self.PLAYER_X_POS + self.PLAYER_WIDTH / 2,
                        self.player_y + self.PLAYER_HEIGHT,
                        color=(200, 200, 220),
                        life=20,
                        vy_range=(1, 4)
                    )

            # Update player physics
            self.player_vy += self.PLAYER_GRAVITY
            self.player_y += self.player_vy
            
            # Ground collision
            if self.player_y >= self.TRACK_Y - self.PLAYER_HEIGHT:
                if not self.on_ground: # Just landed
                    self.player_squash = -8 # Squash effect
                    # Sound: Land
                    # Create landing particles
                    for _ in range(5):
                        self._create_particle(
                            self.PLAYER_X_POS + self.PLAYER_WIDTH / 2,
                            self.player_y + self.PLAYER_HEIGHT,
                            color=(150, 150, 160),
                            life=15,
                            vx_range=(-2, 2)
                        )
                self.player_y = self.TRACK_Y - self.PLAYER_HEIGHT
                self.player_vy = 0
                self.on_ground = True
            
            # Update squash/stretch effect
            self.player_squash *= 0.85
            if abs(self.player_squash) < 0.5:
                self.player_squash = 0

            # Update world scroll
            scroll_speed = 8 + (self.lap - 1) * 2.0
            self.track_progress += scroll_speed
            
            # Update obstacles
            new_obstacles = []
            player_rect = self._get_player_rect()
            for obs in self.obstacles:
                obs['x'] -= scroll_speed
                if obs['x'] > -obs['width']:
                    new_obstacles.append(obs)
                    
                    # Collision check
                    obs_rect = pygame.Rect(obs['x'], obs['y'], obs['width'], obs['height'])
                    if player_rect.colliderect(obs_rect):
                        self.game_over = True
                        reward -= 10 # Collision penalty
                        # Sound: Crash
                        # Create explosion particles
                        for _ in range(50):
                            self._create_particle(
                                player_rect.centerx,
                                player_rect.centery,
                                color=self.COLOR_PLAYER,
                                life=40,
                                vx_range=(-8, 8),
                                vy_range=(-8, 8)
                            )
                        break
                    
                    # Check for clearing an obstacle
                    if obs['id'] not in self.cleared_obstacles and obs['x'] + obs['width'] < self.PLAYER_X_POS:
                        reward += 1
                        self.cleared_obstacles.add(obs['id'])
                        # Sound: Point score

            self.obstacles = new_obstacles

            # Lap completion
            if self.track_progress >= self.TRACK_LENGTH:
                self.lap += 1
                self.track_progress %= self.TRACK_LENGTH
                self.cleared_obstacles.clear()
                if self.lap <= self.MAX_LAPS:
                    reward += 50
                    self._generate_track()
                    # Sound: Lap complete
            
            # Update particles
            self._update_particles(scroll_speed)
            if self.on_ground and self.steps % 3 == 0: # Dust trail
                 self._create_particle(
                    self.PLAYER_X_POS,
                    self.player_y + self.PLAYER_HEIGHT,
                    color=(150, 150, 160),
                    life=25,
                    vx_range=(-3, -1)
                )

        # --- Check termination conditions ---
        if self.game_over:
            terminated = True
        
        if self.lap > self.MAX_LAPS:
            if self.time_elapsed <= self.TIME_LIMIT:
                self.win_condition = True
                reward += 100 # Win bonus
            else:
                reward -= 100 # Lost due to time on the final lap
            self.game_over = True
            terminated = True
        
        if self.time_elapsed >= self.TIME_LIMIT and not self.win_condition:
            reward -= 100 # Time limit loss
            self.game_over = True
            terminated = True

        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self._draw_gradient_background()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": min(self.lap, self.MAX_LAPS),
            "time": self.time_elapsed,
        }

    def _generate_track(self):
        # Procedurally generate obstacles for a lap
        current_x = self.SCREEN_WIDTH + 200 # Start spawning off-screen
        obstacle_id_start = len(self.cleared_obstacles) + len(self.obstacles)
        while current_x < self.TRACK_LENGTH:
            gap = self.np_random.integers(200, 500)
            current_x += gap
            
            width = self.np_random.integers(40, 80)
            height = self.np_random.integers(30, 60)
            
            self.obstacles.append({
                'id': obstacle_id_start + len(self.obstacles),
                'x': current_x,
                'y': self.TRACK_Y - height,
                'width': width,
                'height': height
            })

    def _get_player_rect(self):
        height = self.PLAYER_HEIGHT - self.player_squash
        width = self.PLAYER_WIDTH + self.player_squash
        y_pos = self.player_y + self.player_squash / 2
        return pygame.Rect(self.PLAYER_X_POS, y_pos, width, height)

    def _create_particle(self, x, y, color, life, vx_range=(-1, 1), vy_range=(-1, 1)):
        particle = {
            'x': x,
            'y': y,
            'vx': self.np_random.uniform(vx_range[0], vx_range[1]),
            'vy': self.np_random.uniform(vy_range[0], vy_range[1]),
            'life': life,
            'max_life': life,
            'color': color
        }
        self.particles.append(particle)
        
    def _update_particles(self, scroll_speed):
        active_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['x'] += p['vx'] - scroll_speed
                p['y'] += p['vy']
                p['vy'] += 0.1 # Particle gravity
                active_particles.append(p)
        self.particles = active_particles

    def _draw_gradient_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.TRACK_Y))
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = max(1, int(4 * (p['life'] / p['max_life'])))
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill((*p['color'], alpha))
            self.screen.blit(s, (int(p['x']), int(p['y'])))

        # Draw obstacles
        for obs in self.obstacles:
            if obs['x'] < self.SCREEN_WIDTH and obs['x'] > -obs['width']:
                rect = (int(obs['x']), int(obs['y']), obs['width'], obs['height'])
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, rect, 2)
        
        # Draw player
        if not (self.game_over and not self.win_condition):
            player_rect = self._get_player_rect()
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, 2, border_radius=3)

    def _render_text(self, text, font, x, y, color, shadow_color=None):
        if shadow_color:
            text_surface = font.render(text, True, shadow_color)
            self.screen.blit(text_surface, (x + 2, y + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def _render_ui(self):
        # Render Lap/Time
        time_str = f"TIME: {self.time_elapsed:.2f}"
        lap_str = f"LAP: {min(self.lap, self.MAX_LAPS)} / {self.MAX_LAPS}"
        
        self._render_text(time_str, self.font_medium, 10, 10, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        lap_surf = self.font_medium.render(lap_str, True, self.COLOR_TEXT)
        self._render_text(lap_str, self.font_medium, self.SCREEN_WIDTH - lap_surf.get_width() - 10, 10, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Render Game Over/Win message
        if self.game_over:
            if self.win_condition:
                message = "YOU WIN!"
                color = (100, 255, 100)
            else:
                message = "GAME OVER"
                color = (255, 100, 100)
            
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2 + 4, self.SCREEN_HEIGHT / 2 + 4))
            self.screen.blit(text_surf, text_rect)

            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Requires pygame to be installed with video support
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'quartz', etc.

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen_human = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human input ---
        space_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_pressed = True

        # Convert human input to the MultiDiscrete action format
        # action[0] (movement) is 0 (none)
        # action[1] (space) is 1 if pressed, 0 otherwise
        # action[2] (shift) is 0 (released)
        action = [0, 1 if space_pressed else 0, 0]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation to the human-visible screen ---
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_human.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Time: {info['time']:.2f}s, Laps: {info['lap']-1}")
            # Wait a moment before closing
            pygame.time.wait(3000)

    env.close()