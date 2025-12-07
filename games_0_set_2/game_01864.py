
# Generated: 2025-08-27T18:33:45.329704
# Source Brief: brief_01864.md
# Brief Index: 1864

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame


# Using a simple class for obstacles and particles for clarity
class Obstacle:
    def __init__(self, x, width, height, id):
        self.x = x
        self.width = width
        self.height = height
        self.id = id

class Particle:
    def __init__(self, x, y, vx, vy, size, lifespan, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Press Space to jump. Time your jumps to clear the obstacles."
    )

    game_description = (
        "A minimalist side-scrolling platformer. Time your jumps precisely to overcome "
        "procedurally generated obstacles and reach the end of the final stage."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_Y = self.HEIGHT - 50

        # Game constants
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = -11
        self.PLAYER_X_POS = self.WIDTH // 4
        self.PLAYER_SIZE = 24
        self.MAX_EPISODE_STEPS = 5000
        self.STAGE_LENGTH = 4000
        self.BASE_SCROLL_SPEED = 4
        self.SPEED_INCREASE_PER_STAGE = 1

        # Colors
        self.COLOR_BG_TOP = (50, 50, 120)
        self.COLOR_BG_BOTTOM = (10, 10, 50)
        self.COLOR_GROUND = (60, 60, 70)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 0, 50)
        self.COLOR_OBSTACLE = (220, 50, 50)
        self.COLOR_OBSTACLE_OUTLINE = (150, 20, 20)
        self.COLOR_STAGE_END = (50, 220, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEART = (255, 0, 0)
        
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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_big = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 0
        self.stage = 0
        self.stage_progress = 0
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = False
        self.jump_cooldown = 0
        self.invincibility_timer = 0
        self.stage_clear_timer = 0
        self.obstacles = []
        self.particles = []
        self.cleared_obstacles = set()
        self.last_obstacle_id = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 3
        self.stage = 1
        self.stage_progress = 0
        
        self.player_y = self.GROUND_Y - self.PLAYER_SIZE
        self.player_vy = 0
        self.on_ground = True
        self.jump_cooldown = 0
        self.invincibility_timer = 0
        self.stage_clear_timer = 0
        
        self.particles = []
        self.cleared_obstacles = set()
        self.last_obstacle_id = 0
        self._generate_stage_obstacles()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        terminated = False
        
        if not self.game_over:
            reward = self._update_game_state(action)
            self.score += reward

        terminated = self.game_over or self.steps >= self.MAX_EPISODE_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, action):
        space_held = action[1] == 1
        
        # Pause logic for stage clear transition
        if self.stage_clear_timer > 0:
            self.stage_clear_timer -= 1
            return 0

        # Main game logic
        self.steps += 1
        reward = 0.1 # Small reward for surviving
        
        self._handle_input(space_held)
        self._update_player()
        
        scroll_speed = self.BASE_SCROLL_SPEED + (self.stage - 1) * self.SPEED_INCREASE_PER_STAGE
        self.stage_progress += scroll_speed
        
        obstacle_reward = self._update_obstacles(scroll_speed)
        reward += obstacle_reward
        
        self._update_particles()
        
        collision_reward, did_collide = self._check_collisions()
        reward += collision_reward
        
        if did_collide:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                reward -= 50 # Terminal penalty for losing
        
        # Stage completion
        if self.stage_progress >= self.STAGE_LENGTH and not self.win:
            self.stage += 1
            reward += 50
            if self.stage > 3:
                self.win = True
                self.game_over = True
                reward += 100 # Final win bonus
            else:
                self.stage_progress = 0
                self.cleared_obstacles.clear()
                self.last_obstacle_id = 0
                self._generate_stage_obstacles()
                self.stage_clear_timer = 60 # 2-second pause

        return reward

    def _handle_input(self, space_held):
        if space_held and self.on_ground and self.jump_cooldown == 0:
            self.player_vy = self.JUMP_STRENGTH
            self.on_ground = False
            self.jump_cooldown = 10 # 1/3 second cooldown
            # Sound: Jump
            self._spawn_particles(self.PLAYER_X_POS + self.PLAYER_SIZE/2, self.GROUND_Y, 10, 'jump')

    def _update_player(self):
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
            
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        if not self.on_ground:
            self.player_vy += self.GRAVITY
            self.player_y += self.player_vy

        if self.player_y >= self.GROUND_Y - self.PLAYER_SIZE:
            if not self.on_ground: # Just landed
                # Sound: Land
                self._spawn_particles(self.PLAYER_X_POS + self.PLAYER_SIZE/2, self.GROUND_Y, 15, 'land')
            self.player_y = self.GROUND_Y - self.PLAYER_SIZE
            self.player_vy = 0
            self.on_ground = True

    def _update_obstacles(self, scroll_speed):
        reward = 0
        for obs in self.obstacles:
            obs.x -= scroll_speed
            if obs.x + obs.width < self.PLAYER_X_POS and obs.id not in self.cleared_obstacles:
                reward += 1
                self.cleared_obstacles.add(obs.id)
                # Sound: Obstacle Cleared
        
        self.obstacles = [o for o in self.obstacles if o.x + o.width > 0]
        return reward

    def _generate_stage_obstacles(self):
        self.obstacles.clear()
        current_x = self.WIDTH
        while current_x < self.STAGE_LENGTH:
            gap = self.np_random.integers(150, 300)
            current_x += gap
            width = self.np_random.integers(40, 80)
            max_height = min(150, self.PLAYER_X_POS) # Ensure jump is possible
            height = self.np_random.integers(30, max_height)
            
            self.last_obstacle_id += 1
            self.obstacles.append(Obstacle(current_x, width, height, self.last_obstacle_id))

    def _check_collisions(self):
        if self.invincibility_timer > 0:
            return 0, False
        
        player_rect = pygame.Rect(self.PLAYER_X_POS, self.player_y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs.x, self.GROUND_Y - obs.height, obs.width, obs.height)
            if player_rect.colliderect(obs_rect):
                # Sound: Hit
                self._spawn_particles(player_rect.centerx, player_rect.centery, 20, 'hit')
                self.invincibility_timer = 90 # 3 seconds of invincibility
                return -5, True
        
        return 0, False

    def _spawn_particles(self, x, y, count, p_type):
        for _ in range(count):
            if p_type == 'jump':
                vx = self.np_random.uniform(-1, 1)
                vy = self.np_random.uniform(1, 3)
                size = self.np_random.integers(2, 5)
                lifespan = self.np_random.integers(10, 20)
                color = (180, 180, 180)
            elif p_type == 'land':
                vx = self.np_random.uniform(-2, 2) * (1 if self.np_random.random() > 0.5 else -1)
                vy = self.np_random.uniform(-1, -0.5)
                size = self.np_random.integers(3, 6)
                lifespan = self.np_random.integers(15, 25)
                color = (100, 100, 110)
            elif p_type == 'hit':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                size = self.np_random.integers(3, 7)
                lifespan = self.np_random.integers(20, 40)
                color = self.COLOR_OBSTACLE if self.np_random.random() > 0.3 else self.COLOR_PLAYER
            
            self.particles.append(Particle(x, y, vx, vy, size, lifespan, color))

    def _update_particles(self):
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.lifespan -= 1
            p.size = max(0, p.size - 0.1)
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            # Simple vertical gradient
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Stage End Marker
        end_marker_x = self.STAGE_LENGTH - self.stage_progress
        if 0 < end_marker_x < self.WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_STAGE_END, (end_marker_x, 0, 10, self.GROUND_Y))

        # Obstacles
        for obs in self.obstacles:
            if obs.x < self.WIDTH and obs.x + obs.width > 0:
                obs_rect = pygame.Rect(int(obs.x), int(self.GROUND_Y - obs.height), int(obs.width), int(obs.height))
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obs_rect, 2)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color_with_alpha = p.color + (alpha,)
            s = pygame.Surface((int(p.size)*2, int(p.size)*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (int(p.size), int(p.size)), int(p.size))
            self.screen.blit(s, (int(p.x - p.size), int(p.y - p.size)))

        # Player
        if self.invincibility_timer == 0 or (self.invincibility_timer // 4) % 2 == 0:
            player_rect = pygame.Rect(self.PLAYER_X_POS, int(self.player_y), self.PLAYER_SIZE, self.PLAYER_SIZE)
            
            # Glow effect
            glow_size = int(self.PLAYER_SIZE * 1.8)
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_size//2, glow_size//2, glow_size//2, self.COLOR_PLAYER_GLOW)
            self.screen.blit(glow_surf, (player_rect.centerx - glow_size//2, player_rect.centery - glow_size//2))
            
            # Player body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
    
    def _render_ui(self):
        # Lives
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, 30 + i * 35, 30, 12, self.COLOR_HEART)
            pygame.gfxdraw.aacircle(self.screen, 30 + i * 35, 30, 12, self.COLOR_HEART)

        # Stage
        stage_text = self.font_ui.render(f"Stage: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 20, 20))

        # Game Over / Win Text
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            end_text = self.font_big.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
        elif self.stage_clear_timer > 0:
            msg = "STAGE CLEAR!"
            clear_text = self.font_big.render(msg, True, self.COLOR_TEXT)
            alpha = min(255, int(510 * (1 - abs(self.stage_clear_timer - 30) / 30)))
            clear_text.set_alpha(alpha)
            text_rect = clear_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(clear_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "lives": self.lives,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of the environment's functionality
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'x11', 'dummy' or 'windows'
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    pygame.display.set_caption("Jumper Game")
    game_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop for manual play
    while not terminated:
        # Get player input
        keys = pygame.key.get_pressed()
        space_pressed = keys[pygame.K_SPACE]
        
        # Map input to action space
        # actions[0]: Movement (unused)
        # actions[1]: Space button
        # actions[2]: Shift button (unused)
        action = [0, 1 if space_pressed else 0, 0]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the game window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling for closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        # Optional: Print info for debugging
        # print(f"Score: {info['score']:.1f}, Lives: {info['lives']}, Reward: {reward:.2f}")

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}")