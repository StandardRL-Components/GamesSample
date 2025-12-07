
# Generated: 2025-08-27T21:15:08.230894
# Source Brief: brief_02724.md
# Brief Index: 2724

        
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
        "Controls: ←→ to move. ↑ to jump. Hold Shift while jumping for a higher jump. ↓ to fall faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms, dodging obstacles, to reach the top within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (20, 30, 50)
    COLOR_BG_BOTTOM = (40, 60, 100)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    COLOR_PLATFORM = (100, 110, 130)
    COLOR_PLATFORM_TOP = (200, 210, 230)
    COLOR_OBSTACLE = (255, 165, 0)
    COLOR_OBSTACLE_OUTLINE = (255, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GOAL = (255, 215, 0)
    
    # Physics
    GRAVITY = 0.35
    PLAYER_MOVE_SPEED = 3.0
    SMALL_JUMP_POWER = -8.0
    LARGE_JUMP_POWER = -11.0
    FAST_FALL_SPEED = 2.0
    PLAYER_AIR_CONTROL = 0.5
    MAX_VEL_Y = 10.0
    
    # Game parameters
    MAX_STEPS = 1000 # 10 seconds at 100hz, but step-based means it's just a step limit.
    PLAYER_SIZE = (20, 20)
    PLATFORM_HEIGHT = 15
    GOAL_HEIGHT_Y = 10000
    
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, *self.PLAYER_SIZE)
        self.on_ground = False
        self.last_platform_y = 0
        self.max_y_reached = 0

        self.platforms = []
        self.obstacles = []
        self.particles = []
        
        self.camera_y = 0
        self.highest_platform_y = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_reason = ""
        self.rng = None
        
        # Initialize state variables
        self.reset()

        # This will fail if the implementation is wrong
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_reason = ""
        
        self.camera_y = 0
        self.highest_platform_y = self.SCREEN_HEIGHT - 50

        # Player state
        start_platform_w = 120
        start_platform = pygame.Rect(
            (self.SCREEN_WIDTH - start_platform_w) / 2,
            self.SCREEN_HEIGHT - 50,
            start_platform_w,
            self.PLATFORM_HEIGHT
        )
        self.platforms = [{'rect': start_platform, 'type': 'start', 'passed': False}]
        
        self.player_pos = pygame.math.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE[1])
        self.player_vel = pygame.math.Vector2(0, 0)
        self.on_ground = True
        self.last_platform_y = self.player_pos.y
        self.max_y_reached = self.player_pos.y
        
        # World state
        self.obstacles = []
        self.particles = []
        self._generate_initial_platforms()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            # Unpack factorized action
            movement, _, shift_held_raw = action
            shift_held = shift_held_raw == 1
            
            # --- Update game logic ---
            self.steps += 1
            reward += 0.1 # Survival reward
            
            self._handle_input(movement, shift_held)
            self._update_player()
            collision_reward = self._handle_collisions()
            reward += collision_reward
            self._update_world()
            self._update_particles()

            # --- Check for termination ---
            if self.player_pos.y > self.camera_y + self.SCREEN_HEIGHT + 50:
                self.game_over = True
                self.game_over_reason = "You Fell!"
                reward -= 50
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                self.game_over_reason = "Time's Up!"
                reward -= 10
            
            terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
        
    def _handle_input(self, movement, shift_held):
        # Horizontal movement
        if self.on_ground:
            if movement == 3: # Left
                self.player_vel.x = -self.PLAYER_MOVE_SPEED
            elif movement == 4: # Right
                self.player_vel.x = self.PLAYER_MOVE_SPEED
            else:
                self.player_vel.x = 0
        else: # Air control
            if movement == 3: # Left
                self.player_vel.x -= self.PLAYER_AIR_CONTROL
            elif movement == 4: # Right
                self.player_vel.x += self.PLAYER_AIR_CONTROL
            self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MOVE_SPEED, self.PLAYER_MOVE_SPEED)

        # Jumping
        if movement == 1 and self.on_ground: # Up
            jump_power = self.LARGE_JUMP_POWER if shift_held else self.SMALL_JUMP_POWER
            self.player_vel.y = jump_power
            self.on_ground = False
            # Sound placeholder: # sfx_jump.play()
            self._create_jump_particles(15)

        # Fast fall
        if movement == 2 and not self.on_ground: # Down
            self.player_vel.y = max(self.player_vel.y, self.FAST_FALL_SPEED)

    def _update_player(self):
        # Apply gravity
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, self.MAX_VEL_Y)

        # Update position
        self.player_pos += self.player_vel
        
        # Horizontal screen wrap
        if self.player_pos.x > self.SCREEN_WIDTH:
            self.player_pos.x = 0
        elif self.player_pos.x < 0:
            self.player_pos.x = self.SCREEN_WIDTH
            
        self.player_rect.bottomleft = self.player_pos
        self.max_y_reached = min(self.max_y_reached, self.player_pos.y)

    def _handle_collisions(self):
        reward = 0
        self.on_ground = False
        
        # Player-Platform collision
        for platform_data in self.platforms:
            platform_rect = platform_data['rect']
            if (self.player_vel.y > 0 and 
                platform_rect.colliderect(self.player_rect) and
                self.player_rect.bottom - self.player_vel.y <= platform_rect.top):
                
                self.player_pos.y = platform_rect.top
                self.player_vel.y = 0
                self.player_vel.x *= 0.8 # friction
                self.on_ground = True
                # Sound placeholder: # sfx_land.play()
                self._create_jump_particles(5)

                # Reward for landing on a lower platform
                if self.player_pos.y > self.last_platform_y + self.PLATFORM_HEIGHT * 2:
                    reward -= 0.2
                self.last_platform_y = self.player_pos.y

                # Reward for risky jump
                dist_from_center = abs(self.player_rect.centerx - platform_rect.centerx)
                if dist_from_center > platform_rect.width * 0.4:
                    reward += 5
                    # Sound placeholder: # sfx_risky_land.play()
                
                # Check for goal
                if platform_data['type'] == 'goal':
                    self.game_over = True
                    self.game_over_reason = "You Win!"
                    reward += 100
                break

        # Player-Obstacle collision
        for obstacle in self.obstacles:
            if obstacle['rect'].colliderect(self.player_rect):
                self.game_over = True
                self.game_over_reason = "Hit Obstacle!"
                reward -= 50
                # Sound placeholder: # sfx_hit.play()
                self._create_hit_particles(30)
                break
        
        # Reward for passing obstacles
        for obstacle in self.obstacles:
            if not obstacle['passed'] and self.player_rect.bottom < obstacle['rect'].top:
                obstacle['passed'] = True
                reward += 1
                self.score += 10 # Extra score for dodging
        
        return reward

    def _update_world(self):
        # Update obstacle speed based on steps
        difficulty_multiplier = 1.0 + (self.steps // 500) * 0.05

        # Move obstacles
        for o in self.obstacles:
            o['rect'].x += o['vel_x'] * difficulty_multiplier
            if o['rect'].left < 0 or o['rect'].right > self.SCREEN_WIDTH:
                o['vel_x'] *= -1

        # Scroll camera
        if self.player_pos.y < self.camera_y + self.SCREEN_HEIGHT / 2.5:
            scroll_amount = (self.camera_y + self.SCREEN_HEIGHT / 2.5) - self.player_pos.y
            self.camera_y -= scroll_amount
            self.score += int(scroll_amount) # Score for gaining height

        # Generate new platforms and obstacles
        while self.highest_platform_y > self.camera_y - 50:
            self._generate_platform_and_obstacle()

        # Despawn off-screen entities
        self.platforms = [p for p in self.platforms if p['rect'].top < self.camera_y + self.SCREEN_HEIGHT + 20]
        self.obstacles = [o for o in self.obstacles if o['rect'].top < self.camera_y + self.SCREEN_HEIGHT + 20]

    def _generate_initial_platforms(self):
        for _ in range(20):
            self._generate_platform_and_obstacle()
        
        # Add goal platform
        goal_w = self.SCREEN_WIDTH - 100
        goal_rect = pygame.Rect(50, self.highest_platform_y - self.GOAL_HEIGHT_Y, goal_w, self.PLATFORM_HEIGHT)
        self.platforms.append({'rect': goal_rect, 'type': 'goal', 'passed': False})

    def _generate_platform_and_obstacle(self):
        last_platform = self.platforms[-1]['rect']
        
        min_w, max_w = 80, 150
        min_dx, max_dx = -150, 150
        min_dy, max_dy = 60, 140

        w = self.rng.integers(min_w, max_w)
        dx = self.rng.integers(min_dx, max_dx)
        dy = self.rng.integers(min_dy, max_dy)

        new_x = last_platform.centerx + dx - w / 2
        new_x = np.clip(new_x, 0, self.SCREEN_WIDTH - w)
        new_y = last_platform.y - dy

        new_platform_rect = pygame.Rect(new_x, new_y, w, self.PLATFORM_HEIGHT)
        self.platforms.append({'rect': new_platform_rect, 'type': 'normal', 'passed': False})
        self.highest_platform_y = new_y

        # Chance to spawn an obstacle on the new platform
        if self.rng.random() < 0.4 and new_y < self.SCREEN_HEIGHT - 200:
            obs_radius = 8
            obs_x = new_platform_rect.centerx
            obs_y = new_platform_rect.top - obs_radius
            obs_vel_x = self.rng.choice([-1.5, 1.5])
            self.obstacles.append({
                'rect': pygame.Rect(obs_x - obs_radius, obs_y - obs_radius, obs_radius*2, obs_radius*2), 
                'vel_x': obs_vel_x,
                'passed': False
            })

    def _get_observation(self):
        # Clear screen with background
        self._draw_gradient_background()
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw particles
        for p in self.particles:
            color = p['color']
            alpha = int(255 * (p['life'] / p['max_life']))
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'].x - self.camera_y - p['size']), int(p['pos'].y - self.camera_y - p['size'])))

        # Draw platforms
        for p_data in self.platforms:
            r = p_data['rect']
            draw_r = r.move(0, -self.camera_y)
            color = self.COLOR_GOAL if p_data['type'] == 'goal' else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, draw_r, border_radius=3)
            pygame.draw.line(self.screen, self.COLOR_PLATFORM_TOP, draw_r.topleft, draw_r.topright, 2)

        # Draw obstacles
        for o in self.obstacles:
            r = o['rect']
            draw_r = r.move(0, -self.camera_y)
            pygame.gfxdraw.filled_circle(self.screen, int(draw_r.centerx), int(draw_r.centery), int(r.width/2), self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, int(draw_r.centerx), int(draw_r.centery), int(r.width/2), self.COLOR_OBSTACLE_OUTLINE)

        # Draw player
        draw_player_rect = self.player_rect.move(0, -self.camera_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, draw_player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, draw_player_rect, 1, border_radius=3)
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"Time: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_game_over.render(self.game_over_reason, True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _draw_gradient_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "y_pos": self.player_pos.y,
            "max_y_reached": self.max_y_reached,
        }

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_jump_particles(self, count):
        for _ in range(count):
            self.particles.append({
                'pos': self.player_rect.midbottom,
                'vel': pygame.math.Vector2(self.rng.uniform(-1, 1), self.rng.uniform(0.5, 2)),
                'life': self.rng.integers(10, 25),
                'max_life': 25,
                'size': self.rng.integers(1, 4),
                'color': (200, 200, 255)
            })

    def _create_hit_particles(self, count):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 5)
            self.particles.append({
                'pos': self.player_rect.center,
                'vel': pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.rng.integers(20, 40),
                'max_life': 40,
                'size': self.rng.integers(2, 5),
                'color': self.COLOR_OBSTACLE
            })

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

# Example of how to run the environment
if __name__ == '__main__':
    import time

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Setup ---
    # Set up a window to display the game
    pygame.display.set_caption("Hopper Game")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    total_reward = 0
    
    # Game loop for manual play
    while not done:
        # Get action from keyboard
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            time.sleep(2) # Pause before resetting
            obs, info = env.reset()
            total_reward = 0
        
        # Since auto_advance is False, we control the frame rate here for manual play
        env.clock.tick(60)

    env.close()