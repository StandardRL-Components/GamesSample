
# Generated: 2025-08-28T02:13:54.286533
# Source Brief: brief_04380.md
# Brief Index: 4380

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: ↑/↓ to steer. Dodge the red blocks and reach the finish line before time runs out."
    )

    game_description = (
        "A fast-paced, side-scrolling retro racer. Navigate a procedurally generated "
        "track, avoid obstacles, and race against the clock to reach the finish line."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.FINISH_DISTANCE = 10000 
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors (Retro Neon)
        self.COLOR_BG = (13, 2, 33)
        self.COLOR_TRACK = (48, 25, 52)
        self.COLOR_PLAYER = (0, 255, 136)
        self.COLOR_PLAYER_GLOW = (0, 255, 136, 50)
        self.COLOR_OBSTACLE = (255, 0, 111)
        self.COLOR_FINISH_LINE = (255, 255, 255)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 82, 197)
        
        # Game constants
        self.TRACK_TOP = 50
        self.TRACK_BOTTOM = self.HEIGHT - 50
        self.TRACK_HEIGHT = self.TRACK_BOTTOM - self.TRACK_TOP
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 15, 15
        self.PLAYER_SPEED = 5
        self.PLAYER_X_POS = self.WIDTH // 4
        self.OBSTACLE_SIZE = 20
        self.INITIAL_OBSTACLE_SPEED = 4.0
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.collisions = 0
        self.game_progress = 0
        self.camera_x = 0
        self.player_pos = None
        self.player_vel_y = 0
        self.obstacles = []
        self.particles = []
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.next_obstacle_spawn = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.collisions = 0
        self.game_progress = 0
        self.camera_x = 0
        
        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.HEIGHT / 2)
        self.player_vel_y = 0
        
        self.obstacles = []
        self.particles = []
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.next_obstacle_spawn = self.WIDTH
        
        # Generate initial obstacles
        for i in range(5):
            self._spawn_obstacle(self.WIDTH + i * 300)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        reward = 0
        terminated = False
        
        if not self.game_over:
            self._update_player(movement)
            self._update_world()
            
            collision_this_step = self._handle_collisions()
            
            # Calculate reward
            if collision_this_step:
                reward -= 10
                # sfx: explosion
            else:
                reward += 0.1 # Reward for surviving
                
            # Update score
            self.score += reward

            # Check for termination conditions
            if self.collisions >= 3:
                self.game_over = True
                terminated = True
            
            if self.game_progress >= self.FINISH_DISTANCE:
                self.game_over = True
                self.game_won = True
                terminated = True
                time_bonus_ratio = max(0, self.MAX_STEPS - self.steps) / self.MAX_STEPS
                goal_reward = 100 * time_bonus_ratio
                reward += goal_reward
                self.score += goal_reward
                # sfx: win_jingle
            
            if self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # Update vertical velocity based on action
        if movement == 1: # Up
            self.player_vel_y = -self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_vel_y = self.PLAYER_SPEED
        else: # No-op or other keys
            self.player_vel_y = 0
        
        # Update player position
        self.player_pos.y += self.player_vel_y
        
        # Clamp player to track boundaries
        self.player_pos.y = max(self.TRACK_TOP + self.PLAYER_HEIGHT / 2, 
                                min(self.player_pos.y, self.TRACK_BOTTOM - self.PLAYER_HEIGHT / 2))

    def _update_world(self):
        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed += 0.25

        # Update horizontal progress and camera
        self.game_progress += self.obstacle_speed
        self.camera_x = self.game_progress - self.PLAYER_X_POS
        
        # Update obstacles
        for obstacle in self.obstacles:
            obstacle.x -= self.obstacle_speed
        
        # Remove off-screen obstacles and spawn new ones
        self.obstacles = [obs for obs in self.obstacles if obs.right - self.camera_x > 0]
        if self.game_progress > self.next_obstacle_spawn - self.WIDTH:
            self._spawn_obstacle(self.next_obstacle_spawn)
            self.next_obstacle_spawn += self.np_random.integers(200, 400)

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _spawn_obstacle(self, x_pos):
        y_pos = self.np_random.integers(self.TRACK_TOP, self.TRACK_BOTTOM - self.OBSTACLE_SIZE)
        self.obstacles.append(pygame.Rect(x_pos, y_pos, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE))

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_WIDTH / 2,
                                  self.player_pos.y - self.PLAYER_HEIGHT / 2,
                                  self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        collided_this_step = False
        for obstacle in self.obstacles[:]:
            if player_rect.colliderect(obstacle):
                self.collisions += 1
                collided_this_step = True
                self._create_explosion(self.player_pos)
                self.obstacles.remove(obstacle)
                # sfx: collision_sound
                break # Only one collision per frame
        return collided_this_step

    def _create_explosion(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw track boundaries
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_TOP), (self.WIDTH, self.TRACK_TOP), 3)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_BOTTOM), (self.WIDTH, self.TRACK_BOTTOM), 3)

        # Draw finish line
        finish_x_on_screen = self.FINISH_DISTANCE - self.camera_x
        if 0 < finish_x_on_screen < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, 
                             (finish_x_on_screen, self.TRACK_TOP), 
                             (finish_x_on_screen, self.TRACK_BOTTOM), 5)

        # Draw obstacles
        for obstacle in self.obstacles:
            obs_rect = pygame.Rect(obstacle.x - self.camera_x, obstacle.y, obstacle.width, obstacle.height)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)
            pygame.gfxdraw.rectangle(self.screen, obs_rect, (*self.COLOR_OBSTACLE, 150))


        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            alpha = int(255 * (p['life'] / 30))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)

        # Draw player
        p_x, p_y = int(self.player_pos.x), int(self.player_pos.y)
        
        # Glow effect
        glow_radius = int(self.PLAYER_WIDTH * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, p_x, p_y, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player triangle
        points = [
            (p_x + self.PLAYER_WIDTH, p_y),
            (p_x, p_y - self.PLAYER_HEIGHT / 2),
            (p_x, p_y + self.PLAYER_HEIGHT / 2)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Render collisions
        collision_text = self.font_ui.render(f"HITS: {self.collisions} / 3", True, self.COLOR_UI_TEXT)
        self.screen.blit(collision_text, (10, 10))

        # Render timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Render progress
        progress = min(100, (self.game_progress / self.FINISH_DISTANCE) * 100)
        progress_text = self.font_ui.render(f"PROGRESS: {progress:.0f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (self.WIDTH // 2 - progress_text.get_width() // 2, 10))


        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.game_won:
                end_text = self.font_game_over.render("FINISH!", True, self.COLOR_PLAYER)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_OBSTACLE)
                
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collisions": self.collisions,
            "progress_percent": (self.game_progress / self.FINISH_DISTANCE) * 100,
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        # This is just for human play; RL agents would provide a full action
        action = [movement, 0, 0] 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")
        
        clock.tick(env.FPS)
        
    env.close()