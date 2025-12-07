
# Generated: 2025-08-28T04:15:42.558318
# Source Brief: brief_02253.md
# Brief Index: 2253

        
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
        "Controls: Use ←→ to move on platforms. Press Space to jump. Use ←→ for air control."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap between procedurally generated platforms to reach the top. Don't fall or run out of time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 second timer

        # Colors
        self.COLOR_BG_TOP = (10, 10, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_PLATFORM = (50, 255, 50)
        self.COLOR_PLATFORM_OUTLINE = (30, 150, 30)
        self.COLOR_GOAL_PLATFORM = (255, 215, 0)
        self.COLOR_GOAL_PLATFORM_OUTLINE = (200, 160, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (220, 220, 255)

        # Physics
        self.GRAVITY = 0.6
        self.PLAYER_MOVE_SPEED = 4.0
        self.PLAYER_AIR_CONTROL = 0.4
        self.JUMP_VELOCITY = -12.0
        self.MAX_PLAYER_VEL_X = 6.0

        # Player settings
        self.PLAYER_SIZE = pygame.Vector2(20, 20)
        
        # Platform settings
        self.PLATFORM_HEIGHT = 15
        self.PLATFORM_COUNT = 8
        self.GOAL_JUMP_REQUIREMENT = 20

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Pre-render assets
        self._pre_render_background()
        self._pre_render_glow()
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.platforms = []
        self.particles = []
        self.on_ground = False
        self.jumps_completed = 0
        self.goal_platform_spawned = False
        self.goal_platform_idx = -1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rng = np.random.default_rng()
        
        self.validate_implementation()

    def _pre_render_background(self):
        self.background_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.background_surface, color, (0, y), (self.WIDTH, y))

    def _pre_render_glow(self):
        size = int(self.PLAYER_SIZE.x * 3)
        self.glow_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        for i in range(5):
            alpha = 60 - i * 12
            radius = center * (1 - i / 5.0)
            color = (*self.COLOR_PLAYER_GLOW, alpha)
            pygame.draw.circle(self.glow_surface, color, (center, center), int(radius))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.on_ground = True
        self.platform_speed = 1.0
        self.jumps_completed = 0
        self.goal_platform_spawned = False
        self.goal_platform_idx = -1
        self.particles.clear()
        
        self._generate_platforms(initial=True)
        
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE.y / 2)
        self.player_vel = pygame.Vector2(0, 0)
        
        return self._get_observation(), self._get_info()

    def _generate_platforms(self, initial=False):
        self.platforms.clear()
        plat_y_spacing = self.HEIGHT / (self.PLATFORM_COUNT - 2)
        
        if initial:
            # Create a safe starting platform
            start_width = 120
            start_plat = pygame.Rect(self.WIDTH / 2 - start_width / 2, self.HEIGHT - 40, start_width, self.PLATFORM_HEIGHT)
            self.platforms.append(start_plat)
            
            # Generate subsequent platforms
            for i in range(1, self.PLATFORM_COUNT):
                prev_plat = self.platforms[i-1]
                width = self.rng.integers(80, 140)
                max_reach_x = 180
                x = self.rng.integers(
                    max(0, prev_plat.centerx - max_reach_x),
                    min(self.WIDTH - width, prev_plat.centerx + max_reach_x - width)
                )
                y = prev_plat.y - self.rng.integers(int(plat_y_spacing * 0.8), int(plat_y_spacing * 1.2))
                self.platforms.append(pygame.Rect(x, y, width, self.PLATFORM_HEIGHT))
        else: # For regenerating a single platform
            pass # Handled in _update_platforms

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward
        
        self._handle_input(action)
        self._update_physics()
        self._update_platforms()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self._update_particles()
        self._update_difficulty()
        
        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if self.on_ground:
            if movement == 3: # Left
                self.player_vel.x = -self.PLAYER_MOVE_SPEED
            elif movement == 4: # Right
                self.player_vel.x = self.PLAYER_MOVE_SPEED
            else:
                self.player_vel.x = 0
            
            if space_held:
                self.player_vel.y = self.JUMP_VELOCITY
                self.on_ground = False
                self._create_particles(self.player_pos + pygame.Vector2(0, self.PLAYER_SIZE.y/2), 15, 'jump')
                # sfx: jump
        else: # Air control
            if movement == 3: # Left
                self.player_vel.x -= self.PLAYER_AIR_CONTROL
            elif movement == 4: # Right
                self.player_vel.x += self.PLAYER_AIR_CONTROL
            self.player_vel.x = np.clip(self.player_vel.x, -self.MAX_PLAYER_VEL_X, self.MAX_PLAYER_VEL_X)

    def _update_physics(self):
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel
        else:
            # On a platform, horizontal movement only
            self.player_pos.x += self.player_vel.x

    def _update_platforms(self):
        for i, plat in enumerate(self.platforms):
            plat.y += self.platform_speed
            
            if plat.top > self.HEIGHT:
                # Respawn platform at the top
                is_goal_platform = False
                if self.jumps_completed >= self.GOAL_JUMP_REQUIREMENT and not self.goal_platform_spawned:
                    is_goal_platform = True
                    self.goal_platform_spawned = True
                    self.goal_platform_idx = i
                
                width = 80 if is_goal_platform else self.rng.integers(80, 140)
                plat.width = width
                plat.x = self.rng.integers(0, self.WIDTH - width)
                plat.y = -self.PLATFORM_HEIGHT

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos - self.PLAYER_SIZE / 2, self.PLAYER_SIZE)
        reward = 0
        
        # Keep player on screen horizontally
        if player_rect.left < 0:
            player_rect.left = 0
            self.player_pos.x = player_rect.centerx
            self.player_vel.x = 0
        if player_rect.right > self.WIDTH:
            player_rect.right = self.WIDTH
            self.player_pos.x = player_rect.centerx
            self.player_vel.x = 0
            
        if self.player_vel.y > 0 and not self.on_ground:
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and player_rect.bottom - self.player_vel.y < plat.top:
                    self.on_ground = True
                    self.player_pos.y = plat.top - self.PLAYER_SIZE.y / 2
                    self.player_vel.y = 0
                    self.player_vel.x = 0 # Stop horizontal momentum on land
                    
                    self.jumps_completed += 1
                    reward += 1.0  # Base reward for landing
                    
                    # Check for risky vs safe jump
                    if player_rect.centerx < plat.left or player_rect.centerx > plat.right:
                        reward += 5.0 # Risky jump
                    else:
                        reward -= 0.2 # Safe jump
                        
                    self._create_particles(self.player_pos + pygame.Vector2(0, self.PLAYER_SIZE.y/2), 20, 'land')
                    # sfx: land
                    
                    # Check for goal platform landing
                    if self.goal_platform_spawned and i == self.goal_platform_idx:
                        reward += 100
                        self.game_over = True
                        
                    break
        return reward

    def _check_termination(self):
        if self.game_over: # Win condition handled in collision
            return True
            
        if self.player_pos.y - self.PLAYER_SIZE.y/2 > self.HEIGHT:
            self.game_over = True
            self.score -= 100 # Penalty for falling
            return True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.score -= 100 # Penalty for time out
            return True
        
        return False

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.platform_speed = min(3.0, self.platform_speed + 0.05)
            
    def _create_particles(self, pos, count, p_type):
        for _ in range(count):
            if p_type == 'jump':
                vel = pygame.Vector2(self.rng.uniform(-1, 1), self.rng.uniform(0.5, 2))
            else: # land
                angle = self.rng.uniform(0, 2 * math.pi)
                speed = self.rng.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.rng.integers(15, 30),
                'radius': self.rng.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]
        
    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render platforms
        for i, plat in enumerate(self.platforms):
            is_goal = self.goal_platform_spawned and i == self.goal_platform_idx
            main_color = self.COLOR_GOAL_PLATFORM if is_goal else self.COLOR_PLATFORM
            outline_color = self.COLOR_GOAL_PLATFORM_OUTLINE if is_goal else self.COLOR_PLATFORM_OUTLINE
            
            pygame.draw.rect(self.screen, main_color, plat)
            pygame.draw.rect(self.screen, outline_color, plat, 2)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*self.COLOR_PARTICLE, alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE.x, self.PLAYER_SIZE.y)
        player_rect.center = self.player_pos

        glow_pos = (player_rect.centerx - self.glow_surface.get_width() / 2,
                    player_rect.centery - self.glow_surface.get_height() / 2)
        self.screen.blit(self.glow_surface, glow_pos)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_large.render(f"Time: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WON!" if self.score > 0 else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)

            final_score_text = self.font_small.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "jumps": self.jumps_completed,
            "on_ground": self.on_ground,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Hopper")
    clock = pygame.time.Clock()

    total_reward = 0
    
    print(env.user_guide)

    while not done:
        # Map keyboard keys to MultiDiscrete action
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}")
    env.close()