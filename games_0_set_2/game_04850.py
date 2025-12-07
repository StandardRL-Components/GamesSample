import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode, which is required for server-side execution.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press SPACE to jump over the approaching obstacles. Survive as long as you can!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced side-scrolling arcade game. Survive by hopping over procedurally generated obstacles. The game speeds up and gets harder over three distinct stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Colors
        self.COLOR_BG_TOP = (40, 20, 80)
        self.COLOR_BG_BOTTOM = (80, 40, 120)
        self.COLOR_GROUND = (30, 60, 30)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 0, 50)
        self.COLOR_OBSTACLE_1 = (200, 50, 50)
        self.COLOR_OBSTACLE_2 = (220, 80, 80)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0, 100)

        # Physics
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -14
        self.BASE_SCROLL_SPEED = 4.0
        self.MAX_SCROLL_SPEED = 12.0
        self.SPEED_INCREASE_INTERVAL = 300 # steps (10 seconds at 30fps)
        self.SPEED_INCREASE_AMOUNT = 0.5

        # Player
        self.PLAYER_BASE_WIDTH = 24
        self.PLAYER_BASE_HEIGHT = 36
        self.PLAYER_X_POS = 100
        self.GROUND_Y = self.HEIGHT - 50

        # Stage
        self.STAGE_DURATION = 600 # steps (20 seconds at 30fps)
        self.MAX_STEPS = self.STAGE_DURATION * 3

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel_y = None
        self.player_on_ground = None
        self.player_squash = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_state = None
        self.current_stage = None
        self.scroll_speed = None
        self.obstacle_spawn_timer = None
        self.cleared_obstacles = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.GROUND_Y)
        self.player_vel_y = 0
        self.player_on_ground = True
        self.player_squash = 1.0

        self.obstacles = []
        self.particles = []
        self.cleared_obstacles = set()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        self.current_stage = 1
        self.scroll_speed = self.BASE_SCROLL_SPEED
        self.obstacle_spawn_timer = 60 # Start spawning obstacles after 2 seconds

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        space_held = action[1] == 1
        
        reward = 0.1 # Survival reward
        
        # --- Handle Input ---
        if space_held and self.player_on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.player_on_ground = False
            self.player_squash = 0.7 # Pre-jump squash

        # --- Update Game Logic ---
        self.steps += 1
        
        # Player physics
        self.player_vel_y += self.GRAVITY
        self.player_pos.y += self.player_vel_y
        
        if self.player_pos.y >= self.GROUND_Y:
            if not self.player_on_ground:
                self._create_landing_particles(10)
                self.player_squash = 1.6 # Landing squash
            self.player_pos.y = self.GROUND_Y
            self.player_vel_y = 0
            self.player_on_ground = True
        
        # Player squash/stretch animation
        target_squash = 1.0
        if not self.player_on_ground:
            stretch = abs(self.player_vel_y) * 0.015
            target_squash = 1.0 - stretch
        self.player_squash += (target_squash - self.player_squash) * 0.15

        # Update difficulty and stage
        if self.steps % self.SPEED_INCREASE_INTERVAL == 0 and self.steps > 0:
            self.scroll_speed = min(self.MAX_SCROLL_SPEED, self.scroll_speed + self.SPEED_INCREASE_AMOUNT)

        if self.steps > 0 and self.steps % self.STAGE_DURATION == 0:
            if self.current_stage < 3:
                self.current_stage += 1
                reward += 100
                self.obstacles.clear()
                self.cleared_obstacles.clear()
                self.obstacle_spawn_timer = 90 # Pause for stage transition
            elif not self.win_state: # Just completed stage 3
                self.win_state = True
                self.game_over = True
                reward += 100 + 300 # Stage 3 clear + game win bonus

        # Obstacle management
        self._update_obstacles()
        self._spawn_obstacles()
        
        # Particle management
        self._update_particles()
        
        # --- Collision and Reward Calculation ---
        player_rect = self._get_player_rect()
        terminated = False

        for obs in self.obstacles:
            # Reward for clearing
            if obs['id'] not in self.cleared_obstacles and obs['rect'].centerx < self.player_pos.x:
                reward += 1
                self.score += 1
                self.cleared_obstacles.add(obs['id'])
            
            # Collision check
            if player_rect.colliderect(obs['rect']):
                self.game_over = True
                reward = -100
                self._create_landing_particles(30, self.COLOR_OBSTACLE_1)
                break
        
        if self.steps >= self.MAX_STEPS and not self.game_over:
             if not self.win_state:
                self.win_state = True
                reward += 100 + 300
             self.game_over = True
        
        if self.game_over:
            terminated = True
        
        # The total score for display includes the survival reward
        display_score_increment = reward if reward > 0 else 0.1
        self.score += display_score_increment
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always False
            self._get_info()
        )

    def _get_player_rect(self):
        w = self.PLAYER_BASE_WIDTH * self.player_squash
        h = self.PLAYER_BASE_HEIGHT / self.player_squash
        return pygame.Rect(self.player_pos.x - w / 2, self.player_pos.y - h, w, h)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['rect'].x -= self.scroll_speed
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

    def _spawn_obstacles(self):
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            width = self.np_random.integers(30, 60)
            
            if self.current_stage == 1:
                height = self.np_random.integers(20, 50)
                self.obstacle_spawn_timer = self.np_random.integers(70, 100)
            elif self.current_stage == 2:
                height = self.np_random.integers(40, 90)
                self.obstacle_spawn_timer = self.np_random.integers(60, 90)
            else: # Stage 3
                height = self.np_random.integers(30, 110)
                self.obstacle_spawn_timer = self.np_random.integers(50, 80)
                # Small chance for a double obstacle
                if self.np_random.random() < 0.2:
                    self._spawn_single_obstacle(width, height)
                    self.obstacle_spawn_timer = self.np_random.integers(20, 30)

            self._spawn_single_obstacle(width, height)
            
            # Adjust timer based on scroll speed to maintain density
            self.obstacle_spawn_timer *= (self.BASE_SCROLL_SPEED / self.scroll_speed)

    def _spawn_single_obstacle(self, width, height):
        new_id = self.steps + self.np_random.integers(1000)
        rect = pygame.Rect(self.WIDTH, self.GROUND_Y - height, width, height)
        self.obstacles.append({'rect': rect, 'id': new_id})

    def _create_landing_particles(self, count, color=None):
        if color is None:
            color = self.COLOR_GROUND
        for _ in range(count):
            angle = self.np_random.uniform(math.pi * 1.1, math.pi * 1.9)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            # FIX: pygame.Vector2 does not have a .copy() method.
            # Create a new vector by passing the original to the constructor,
            # or simply perform the operation, which creates a new vector.
            pos = self.player_pos + pygame.Vector2(0, -5)
            self.particles.append({'pos': pos, 'vel': vel, 'life': self.np_random.integers(20, 40), 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.2 # Particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame's surfarray is (width, height, 3), but gym expects (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 40.0))
            size = max(1, int(p['life'] / 8))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (size, size), size)
            self.screen.blit(s, (int(p['pos'].x - size), int(p['pos'].y - size)))

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_1, obs['rect'])
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_2, obs['rect'].inflate(-6, -6))

        # Player
        if not (self.game_over and not self.win_state):
            player_rect = self._get_player_rect()
            
            # Shadow
            shadow_width = player_rect.width * 0.8
            shadow_height = 8
            shadow_y = self.GROUND_Y - shadow_height / 2
            dist_to_ground = max(0, self.GROUND_Y - player_rect.bottom)
            shadow_alpha = max(0, 100 - dist_to_ground * 0.5)
            shadow_surface = pygame.Surface((shadow_width, shadow_height), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surface, (0,0,0,shadow_alpha), (0,0,shadow_width,shadow_height))
            self.screen.blit(shadow_surface, (player_rect.centerx - shadow_width/2, shadow_y))
            
            # Glow
            glow_radius = int(player_rect.height * 0.8)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
            self.screen.blit(s, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Core
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
            
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.current_stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if self.win_state:
                msg = "YOU WIN!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_OBSTACLE_1
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            shadow = self.font_large.render(msg, True, (0,0,0))
            self.screen.blit(shadow, (text_rect.x+3, text_rect.y+3))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "game_over": self.game_over,
            "win": self.win_state,
        }
        
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It requires a display, so it will not run in the headless test environment
    
    # Re-enable the video driver for local play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hopper Game")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    while not done:
        # Action defaults
        movement, space, shift = 0, 0, 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space = 1
        
        # Construct the action
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    pygame.quit()