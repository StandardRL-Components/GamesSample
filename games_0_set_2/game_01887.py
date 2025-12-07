
# Generated: 2025-08-28T03:01:22.784586
# Source Brief: brief_01887.md
# Brief Index: 1887

        
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

    user_guide = (
        "Hold Space for a small jump or Shift for a high jump to clear the red obstacles."
    )

    game_description = (
        "Control a robot running through a procedurally generated obstacle course. "
        "Jump to avoid obstacles and reach the finish line as fast as possible."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.COURSE_LENGTH = 10000 # Logical length of the course
        self.MAX_STEPS = 1500 # Corresponds to 50 seconds at 30fps

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG_TOP = (100, 120, 200)
        self.COLOR_BG_BOTTOM = (40, 50, 100)
        self.COLOR_GROUND = (60, 160, 80)
        self.COLOR_GROUND_TOP = (80, 190, 100)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 220, 255)
        self.OBSTACLE_COLORS = [(255, 80, 80), (255, 140, 80), (255, 200, 80)]
        self.COLOR_TEXT = (255, 255, 255)
        self.PARTICLE_COLOR = (200, 200, 220)

        # Physics and game parameters
        self.GRAVITY = 0.7
        self.GAME_SPEED = 7
        self.JUMP_SMALL = -11
        self.JUMP_LARGE = -15

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.scroll_x = None
        self.obstacles = None
        self.particles = None
        self.next_obstacle_x = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.rng = None
        
        # Pre-render background for performance
        self.background_surface = self._create_gradient_background()

        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = pygame.Vector2(self.WIDTH * 0.2, self.HEIGHT * 0.75)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        
        self.scroll_x = 0
        self.obstacles = []
        self.particles = []
        self.next_obstacle_x = self.WIDTH * 1.5
        
        # Initial obstacle generation
        for _ in range(5):
            self._generate_obstacle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            # Unpack action
            space_held = action[1] == 1
            shift_held = action[2] == 1

            # --- Player Logic ---
            if self.on_ground:
                if shift_held:
                    self.player_vel.y = self.JUMP_LARGE
                    self.on_ground = False
                    # sfx: large_jump_sound
                elif space_held:
                    self.player_vel.y = self.JUMP_SMALL
                    self.on_ground = False
                    # sfx: small_jump_sound

            # Apply gravity
            self.player_vel.y += self.GRAVITY
            self.player_pos.y += self.player_vel.y

            # Ground collision
            ground_y = self.HEIGHT * 0.75
            if self.player_pos.y > ground_y:
                if not self.on_ground:
                    # Create landing particles
                    for _ in range(10):
                        self.particles.append(self._create_particle(self.player_pos.x, ground_y + 20))
                    # sfx: land_sound
                self.player_pos.y = ground_y
                self.player_vel.y = 0
                self.on_ground = True

            # --- World Logic ---
            self.scroll_x += self.GAME_SPEED
            
            # Update obstacles
            player_rect = self._get_player_rect()
            for obs in self.obstacles:
                # Check for collision
                if player_rect.colliderect(obs['rect']):
                    self.game_over = True
                    terminated = True
                    # sfx: collision_sound
                    break
                
                # Reward for passing an obstacle
                if not obs['cleared'] and obs['rect'].right < player_rect.left:
                    obs['cleared'] = True
                    self.score += 2
                    reward += 2.0
            
            # Remove off-screen obstacles
            self.obstacles = [obs for obs in self.obstacles if obs['rect'].right - self.scroll_x > -50]
            
            # Generate new obstacles
            if self.scroll_x > self.next_obstacle_x - self.WIDTH * 1.5:
                self._generate_obstacle()

            # Update particles
            self._update_particles()
            
            # --- Termination and Score ---
            # Survival reward
            reward += 0.1
            self.score += 0.1

            # Check for win condition
            if self.scroll_x >= self.COURSE_LENGTH:
                self.game_over = True
                self.game_won = True
                terminated = True
                # Calculate time bonus
                time_bonus_max_steps = 45 * 30 # 45s at 30fps
                if self.steps < time_bonus_max_steps:
                    scaling = (time_bonus_max_steps - self.steps) / time_bonus_max_steps
                    win_bonus = 100 * scaling
                    reward += win_bonus
                    self.score += win_bonus
                # sfx: win_sound

        # Check for step limit
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )
        
    def _generate_obstacle(self):
        difficulty_modifier = 1.0 - (0.05 * (self.steps // 100))
        difficulty_modifier = max(0.5, difficulty_modifier) # Cap difficulty

        min_gap = 200 * difficulty_modifier
        max_gap = 400 * difficulty_modifier
        gap = self.rng.uniform(min_gap, max_gap)
        
        width = self.rng.integers(40, 80)
        max_height = 100
        height = self.rng.integers(30, max_height)
        
        ground_y = self.HEIGHT * 0.75
        
        x_pos = self.next_obstacle_x
        y_pos = ground_y - height + 5 # +5 to embed in ground slightly
        
        self.obstacles.append({
            'rect': pygame.Rect(x_pos, y_pos, width, height),
            'color': random.choice(self.OBSTACLE_COLORS),
            'cleared': False
        })
        
        self.next_obstacle_x += width + gap

    def _create_particle(self, x, y):
        return {
            'pos': pygame.Vector2(x, y),
            'vel': pygame.Vector2(self.rng.uniform(-2, 2), self.rng.uniform(-3, -1)),
            'life': self.rng.integers(15, 30),
            'radius': self.rng.uniform(2, 5)
        }

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.2 # Particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 40, 30, 40)

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_game(self):
        # Draw ground
        ground_y = self.HEIGHT * 0.75
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, ground_y, self.WIDTH, self.HEIGHT - ground_y))
        pygame.draw.rect(self.screen, self.COLOR_GROUND_TOP, (0, ground_y, self.WIDTH, 5))

        # Draw obstacles
        for obs in self.obstacles:
            screen_rect = obs['rect'].copy()
            screen_rect.x -= self.scroll_x
            # Simple 3D effect
            darker_color = tuple(max(0, c - 40) for c in obs['color'])
            pygame.draw.rect(self.screen, darker_color, screen_rect.move(5, 5))
            pygame.draw.rect(self.screen, obs['color'], screen_rect)

        # Draw particles
        for p in self.particles:
            pos = p['pos'].copy()
            pos.x -= self.scroll_x
            pygame.draw.circle(self.screen, self.PARTICLE_COLOR, (int(pos.x), int(pos.y)), int(p['radius']))

        # Draw player
        self._render_player()

    def _render_player(self):
        player_rect = self._get_player_rect()
        
        # Animation
        body_bob = math.sin(self.steps * 0.5) * 2 if self.on_ground else 0
        
        # Glow effect
        glow_rect = player_rect.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, self.COLOR_PLAYER_GLOW + (50,), (0, 0, glow_rect.width, glow_rect.height))
        self.screen.blit(glow_surface, glow_rect.topleft)

        # Body
        body_rect = pygame.Rect(player_rect.x, player_rect.y + body_bob, player_rect.width, player_rect.height)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=5)
        
        # Head
        head_pos = (body_rect.centerx, body_rect.top - 10)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, head_pos, 12)
        
        # Eye
        eye_pos = (head_pos[0] + 5, head_pos[1])
        pygame.draw.circle(self.screen, (255, 255, 255), eye_pos, 4)
        pygame.draw.circle(self.screen, (0, 0, 0), eye_pos, 2)


    def _render_ui(self):
        # Timer
        time_elapsed = self.steps / 30.0
        timer_text = f"Time: {time_elapsed:.2f}s"
        text_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 40))

        # Progress bar
        progress = min(1.0, self.scroll_x / self.COURSE_LENGTH)
        bar_width = self.WIDTH - 20
        bar_height = 10
        pygame.draw.rect(self.screen, (50, 50, 80), (10, self.HEIGHT - 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.HEIGHT - 20, bar_width * progress, bar_height))

        # Game Over / Win message
        if self.game_over:
            message = "COURSE COMPLETE!" if self.game_won else "GAME OVER"
            color = (150, 255, 150) if self.game_won else (255, 150, 150)
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Text shadow
            shadow_surf = self.font_game_over.render(message, True, (0,0,0,100))
            self.screen.blit(shadow_surf, text_rect.move(3, 3))
            self.screen.blit(text_surf, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress": self.scroll_x / self.COURSE_LENGTH,
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy", "windib", etc. based on your system
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To run, you need to have pygame installed and a display available.
    # Set the environment variable above if running headlessly.
    
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Pixel Runner")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample() # Start with a random action
    action = [0, 0, 0] # Or a no-op

    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0] # No movement, no buttons
        
        # Map keys to actions
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control FPS
        env.clock.tick(30)

    env.close()
    print("Game Over!")
    print(f"Final Info: {info}")