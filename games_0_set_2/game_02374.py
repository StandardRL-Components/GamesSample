
# Generated: 2025-08-27T20:10:13.942417
# Source Brief: brief_02374.md
# Brief Index: 2374

        
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
        "Controls: Press space to jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer where you jump through a procedurally generated "
        "obstacle course to reach the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Pygame setup
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 255, 50)
        self.COLOR_PLATFORM = (100, 100, 110)
        self.COLOR_PLATFORM_EDGE = (255, 80, 80)
        self.COLOR_FINISH = (80, 255, 80)
        self.COLOR_PARTICLE = (220, 220, 220)
        self.COLOR_UI_TEXT = (200, 200, 220)

        # Fonts
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)

        # Game physics constants
        self.gravity = 0.5
        self.jump_strength = -11
        self.player_size = 20
        self.time_limit_seconds = 30
        self.fps = 30
        self.time_limit_steps = self.time_limit_seconds * self.fps
        self.max_episode_steps = 1000 # Hard cap

        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel_y = None
        self.on_ground = None
        self.platforms = None
        self.finish_line_x = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.prev_space_held = None
        self.last_platform_landed = None

        # Initialize state and validate
        self.reset()
        self.validate_implementation()
    
    def _generate_level(self):
        """Procedurally generates platforms for the level."""
        self.platforms = []
        
        # Starting platform
        start_platform = pygame.Rect(30, self.screen_height - 50, 100, 50)
        self.platforms.append(start_platform)
        
        current_x = start_platform.right
        current_y = start_platform.y
        
        while current_x < self.screen_width - 80:
            # Difficulty scales with horizontal position
            progress = current_x / (self.screen_width - 80)
            
            # Gap distance increases with progress
            min_gap = 30
            max_gap = 80
            gap = min_gap + (max_gap - min_gap) * progress * self.np_random.uniform(0.8, 1.2)
            current_x += gap
            
            # Platform width
            width = self.np_random.integers(60, 150)
            
            # Platform height varies more with progress
            max_y_diff = 10 + 100 * progress
            y_change = self.np_random.uniform(-max_y_diff, max_y_diff)
            new_y = np.clip(current_y + y_change, 150, self.screen_height - 50)
            
            platform_rect = pygame.Rect(int(current_x), int(new_y), int(width), int(self.screen_height - new_y))
            self.platforms.append(platform_rect)
            
            current_x = platform_rect.right
            current_y = new_y
            
        self.finish_line_x = self.screen_width - 40

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()
        
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - self.player_size)
        self.player_vel_y = 0
        self.on_ground = True
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.prev_space_held = False
        self.last_platform_landed = start_platform
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # --- Handle Input ---
        space_held = action[1] == 1
        jumped = False
        if space_held and not self.prev_space_held and self.on_ground:
            self.player_vel_y = self.jump_strength
            self.on_ground = False
            jumped = True
            # sfx: jump_sound()
        self.prev_space_held = space_held

        # --- Physics Update ---
        self.player_vel_y += self.gravity
        # Clamp velocity to prevent extreme speeds
        self.player_vel_y = min(self.player_vel_y, 15)
        self.player_pos.y += self.player_vel_y
        
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.player_size, self.player_size)
        
        # --- Collision Detection ---
        self.on_ground = False
        if self.player_vel_y >= 0: # Only check for landing if moving downwards
            for plat in self.platforms:
                # Check if player's bottom edge is intersecting the platform's top edge
                if player_rect.colliderect(plat) and player_rect.bottom <= plat.top + self.player_vel_y + 1:
                    player_rect.bottom = plat.top
                    self.player_pos.y = player_rect.y
                    self.player_vel_y = 0
                    self.on_ground = True
                    
                    if plat != self.last_platform_landed:
                        self.last_platform_landed = plat
                        reward += 1.0  # Reward for landing on a new platform
                        self.score += 10
                        self._spawn_particles(player_rect.midbottom, 15)
                        # sfx: land_sound()
                    break

        # --- Survival Reward ---
        reward += 0.1
        
        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos.y > self.screen_height:
            terminated = True
            reward = -100.0
            self.score -= 50
            self.game_over = True
            # sfx: fall_sfx()
            
        if player_rect.right >= self.finish_line_x:
            terminated = True
            reward = 100.0
            self.score += 1000
            self.game_over = True
            self.game_won = True
            self._spawn_particles((self.finish_line_x, player_rect.centery), 50)
            # sfx: win_jingle()

        if self.steps >= self.time_limit_steps:
            if not terminated: # Avoid overwriting win/loss reward
                terminated = True
                reward = -100.0
                self.game_over = True
                # sfx: time_out_sfx()

        if self.steps >= self.max_episode_steps:
             if not terminated:
                terminated = True
                reward = -100.0 # Penalize for not finishing
                self.game_over = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_particles(self, position, count):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(position),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0)),
                'radius': self.np_random.uniform(2, 5),
                'lifetime': self.np_random.integers(15, 30),
                'color': self.COLOR_PARTICLE
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Particles have slight gravity
            p['lifetime'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw finish line
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0), (self.finish_line_x, self.screen_height), 3)
        
        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            pygame.draw.line(self.screen, self.COLOR_PLATFORM_EDGE, plat.topleft, plat.topright, 2)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], pos, radius)

        # Draw player
        player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), self.player_size, self.player_size)
        
        # Glow effect
        glow_surf = pygame.Surface((self.player_size * 2, self.player_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.player_size, self.player_size), self.player_size)
        self.screen.blit(glow_surf, (player_rect.centerx - self.player_size, player_rect.centery - self.player_size))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        # Timer
        time_left = max(0, self.time_limit_seconds - (self.steps / self.fps))
        timer_text = f"TIME: {time_left:.1f}"
        timer_surface = self.font_large.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surface, (10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surface = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surface, (self.screen_width - score_surface.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            msg = "VICTORY!" if self.game_won else "GAME OVER"
            color = self.COLOR_FINISH if self.game_won else self.COLOR_PLATFORM_EDGE
            end_surface = self.font_large.render(msg, True, color)
            end_rect = end_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(end_surface, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.time_limit_steps - self.steps)
        }
        
    def close(self):
        pygame.font.quit()
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
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a simple human player loop
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    while not terminated:
        # Action defaults
        movement = 0 # none
        space_btn = 0 # released
        shift_btn = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_btn = 1
        
        # The game is auto-advancing, so we always step
        action = [movement, space_btn, shift_btn]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds before closing
            
        clock.tick(env.fps) # Control the frame rate
        
    env.close()