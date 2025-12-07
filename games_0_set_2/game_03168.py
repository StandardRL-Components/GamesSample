
# Generated: 2025-08-27T22:34:40.863667
# Source Brief: brief_03168.md
# Brief Index: 3168

        
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
        "Controls: Press Space for a short jump or Shift for a long jump. Avoid the red blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling auto-runner. Navigate a procedurally generated slope, "
        "jumping over obstacles to reach the end of each level."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 350
    GRAVITY = 0.6
    MAX_STEPS = 2500

    # Colors
    COLOR_BG_TOP = (20, 30, 50)
    COLOR_BG_BOTTOM = (40, 60, 90)
    COLOR_GROUND = (60, 80, 110)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 150, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (200, 200, 220)

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables (will be properly set in reset)
        self.player_x = 0
        self.player_y = 0
        self.player_y_vel = 0
        self.on_ground = False
        self.last_space_held = False
        self.last_shift_held = False
        self.obstacles = []
        self.particles = []
        self.cleared_obstacles = set()
        self.level_transition_timer = 0
        
        # Initialize state via reset
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.lives = 3
        self.level = 1
        self.game_over = False
        
        self.player_x = 100
        self.player_y = self.GROUND_Y
        self.player_y_vel = 0
        self.on_ground = True
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles.clear()
        self.cleared_obstacles.clear()
        self.level_transition_timer = 0
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # Unpack factorized action
        # movement = action[0] # Unused as per design brief
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- 1. Handle Player Jump Action ---
        jump_initiated = (space_held and not self.last_space_held) or \
                         (shift_held and not self.last_shift_held)

        if jump_initiated and self.on_ground:
            if shift_held: # Long jump (Shift overrides Space)
                self.player_y_vel = -14 # sfx: long_jump
            else: # Short jump
                self.player_y_vel = -10 # sfx: short_jump
            self.on_ground = False
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # --- 2. Update Game Physics ---
        # Player physics
        if not self.on_ground:
            self.player_y_vel += self.GRAVITY
            self.player_y += self.player_y_vel
        
        if self.player_y >= self.GROUND_Y:
            if not self.on_ground: # Just landed
                self._create_particles(self.player_x, self.GROUND_Y, 15, is_landing=True) # sfx: land
            self.player_y = self.GROUND_Y
            self.player_y_vel = 0
            self.on_ground = True

        # Obstacle physics (scrolling)
        scroll_speed = 4.0 + self.level * 0.8
        for obstacle in self.obstacles:
            obstacle.x -= scroll_speed
        
        self.obstacles = [obs for obs in self.obstacles if obs.right > 0]

        # Particle physics
        self._update_particles()
        
        # --- 3. Check for Events & Calculate Rewards ---
        # Survival reward
        reward += 0.01

        player_rect = self._get_player_rect()

        # Collision and jump-clear checks
        obstacles_to_remove = []
        for i, obstacle in enumerate(self.obstacles):
            # Collision check
            if player_rect.colliderect(obstacle):
                reward -= 5.0
                self.lives -= 1
                obstacles_to_remove.append(i)
                self._create_particles(player_rect.centerx, player_rect.centery, 30) # sfx: collision
                break # Only one collision per frame
            
            # Jump clear check
            if i not in self.cleared_obstacles and obstacle.right < self.player_x:
                reward += 1.0
                self.cleared_obstacles.add(i)
                # sfx: clear_obstacle

        # Remove collided obstacles
        if obstacles_to_remove:
            self.obstacles = [obs for i, obs in enumerate(self.obstacles) if i not in obstacles_to_remove]
            # Reset cleared set since indices are now invalid
            self.cleared_obstacles.clear()
            for i, obs in enumerate(self.obstacles):
                if obs.right < self.player_x:
                    self.cleared_obstacles.add(i)

        # Level completion check
        if not self.obstacles and self.level_transition_timer == 0:
            reward += 10.0
            self.level += 1
            if self.level > 5:
                reward += 100.0
                self.game_over = True
            else:
                self.level_transition_timer = 90 # 3 seconds at 30fps
                # sfx: level_complete

        if self.level_transition_timer > 0:
            self.level_transition_timer -= 1
            if self.level_transition_timer == 0 and not self.game_over:
                self._generate_level()

        # --- 4. Update State and Check Termination ---
        self.steps += 1
        self.score += reward
        
        terminated = self.lives <= 0 or self.game_over or self.steps >= self.MAX_STEPS
        
        # Edge case: Player jumps off-screen
        if self.player_y < -50:
            reward -= 10.0
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.obstacles.clear()
        self.cleared_obstacles.clear()
        
        level_length = 3000
        obstacle_min_gap = 250
        obstacle_max_gap = 500
        
        current_x = self.SCREEN_WIDTH + 200
        
        while current_x < level_length:
            gap = self.np_random.integers(obstacle_min_gap, obstacle_max_gap)
            current_x += gap
            
            width = self.np_random.integers(30, 60)
            height = self.np_random.integers(30, 80)
            
            obstacle_rect = pygame.Rect(current_x, self.GROUND_Y - height, width, height)
            self.obstacles.append(obstacle_rect)

    def _get_player_rect(self):
        # A smaller, more accurate hitbox for the triangle
        return pygame.Rect(self.player_x - 10, self.player_y - 28, 20, 28)

    def _create_particles(self, x, y, count, is_landing=False):
        for _ in range(count):
            if is_landing:
                angle = self.np_random.uniform(math.pi * 1.1, math.pi * 1.9)
                speed = self.np_random.uniform(1, 4)
            else: # Explosion
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 6)
                
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': [x, y], 'vel': [vel_x, vel_y], 'life': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # --- Render Game Elements ---
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        # Obstacles
        for obstacle in self.obstacles:
            # Glow effect
            glow_rect = obstacle.inflate(8, 8)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.COLOR_OBSTACLE_GLOW, 60), glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, glow_rect.topleft)
            # Main obstacle
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle, border_radius=3)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30))))
            size = max(1, int(3 * (p['life'] / 30)))
            pygame.draw.circle(self.screen, (*self.COLOR_PARTICLE, alpha), (int(p['pos'][0]), int(p['pos'][1])), size)

        # Player
        player_points = [
            (self.player_x, self.player_y - 30),
            (self.player_x - 15, self.player_y),
            (self.player_x + 15, self.player_y),
        ]
        # Glow effect
        pygame.gfxdraw.aapolygon(self.screen, player_points, (*self.COLOR_PLAYER_GLOW, 100))
        pygame.gfxdraw.filled_polygon(self.screen, player_points, (*self.COLOR_PLAYER_GLOW, 100))
        # Main player triangle
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
        
        # --- Render UI Overlay ---
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        level_text = self.font_main.render(f"LEVEL: {self.level}/5", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (15, 50))
        
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 15, 15))
        
        # Level transition message
        if self.level_transition_timer > 0:
            if self.game_over:
                msg = "YOU WIN!"
            else:
                msg = f"LEVEL {self.level-1} COMPLETE!"
            
            level_msg_surf = self.font_main.render(msg, True, self.COLOR_PLAYER)
            pos = (self.SCREEN_WIDTH // 2 - level_msg_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - 50)
            self.screen.blit(level_msg_surf, pos)
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.level
        }
    
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    # --- Game Loop ---
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # Not used
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))

        # --- Game Over Message ---
        if terminated:
            font = pygame.font.Font(None, 74)
            text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 40))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.Font(None, 36)
            text_restart = font_small.render("Press 'R' to Restart", True, (255, 255, 255))
            text_restart_rect = text_restart.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 20))
            screen.blit(text_restart, text_restart_rect)
            
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False

        pygame.display.flip()
        
        # --- Frame Rate Control ---
        clock.tick(30) # Run at 30 FPS
        
    env.close()