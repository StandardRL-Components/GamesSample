
# Generated: 2025-08-28T00:25:21.335148
# Source Brief: brief_03784.md
# Brief Index: 3784

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A single spark particle for the collision effect."""
    def __init__(self, x, y, rng):
        self.x = x
        self.y = y
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(2, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = rng.integers(25, 45)  # Lifespan in frames
        self.color = (255, rng.integers(100, 200), 0)  # Orange/yellow sparks
        self.radius = self.life / 5

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.radius = max(0, self.life / 5)

    def draw(self, surface):
        if self.life > 0:
            # Use gfxdraw for a smoother circle
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to steer your line. Avoid the red obstacles."
    )

    game_description = (
        "Steer a speeding line through a procedurally generated obstacle course to reach the finish line as quickly as possible."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1800  # 60 seconds at 30fps

    # Player
    PLAYER_WIDTH = 8
    PLAYER_HEIGHT = 40
    PLAYER_Y_POS = 340
    PLAYER_SPEED = 10

    # Track
    TRACK_WIDTH = 300
    TRACK_LEFT = (WIDTH - TRACK_WIDTH) // 2
    TRACK_RIGHT = TRACK_LEFT + TRACK_WIDTH
    TRACK_LINE_WIDTH = 5
    TOTAL_DISTANCE = 40000

    # Obstacles
    OBSTACLE_HEIGHT = 20
    OBSTACLE_ROWS = 4
    OBSTACLE_V_SPACING = 200
    OBSTACLE_GAP_WIDTH = 90 # Player needs to fit through this

    # Physics
    BASE_SCROLL_SPEED = 4.0
    SPEED_INCREMENT_PER_STEP = 0.003

    # Colors (Tron-style)
    COLOR_BG = (10, 15, 30)
    COLOR_TRACK = (80, 100, 120)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 80, 120)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (150, 20, 20)
    COLOR_FINISH = (50, 255, 50)
    COLOR_FINISH_GLOW = (20, 150, 20)
    COLOR_TEXT = (220, 220, 255)
    # ---

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.player_x = 0
        self.obstacles = []
        self.scroll_speed = 0
        self.distance_traveled = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.speed_lines = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_x = self.WIDTH / 2
        self.scroll_speed = self.BASE_SCROLL_SPEED
        self.distance_traveled = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []

        self.obstacles = []
        for i in range(self.OBSTACLE_ROWS):
            self._generate_obstacle_row(initial_y=self.HEIGHT - (i + 1) * self.OBSTACLE_V_SPACING)

        self.speed_lines = []
        for _ in range(70):
            self.speed_lines.append({
                'x': self.np_random.uniform(0, self.WIDTH),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'speed_mult': self.np_random.uniform(1.5, 3.0),
                'height': self.np_random.uniform(20, 60),
                'color': (20, 30, self.np_random.integers(60, 90))
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        if not self.game_over:
            # 1. Update Player Position
            if movement == 3:  # Left
                self.player_x -= self.PLAYER_SPEED
            elif movement == 4:  # Right
                self.player_x += self.PLAYER_SPEED
            self.player_x = np.clip(self.player_x, self.TRACK_LEFT, self.TRACK_RIGHT)

            # 2. Update Game State
            self.steps += 1
            self.scroll_speed += self.SPEED_INCREMENT_PER_STEP
            self.distance_traveled += self.scroll_speed

            # 3. Update Obstacles
            for obs_row in self.obstacles:
                obs_row['y'] += self.scroll_speed
                if obs_row['y'] > self.HEIGHT + self.OBSTACLE_HEIGHT:
                    self._generate_obstacle_row(existing_row=obs_row)
        
        # 4. Update effects (particles, speed lines)
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)
        
        for line in self.speed_lines:
            line['y'] += self.scroll_speed * line['speed_mult']
            if line['y'] > self.HEIGHT:
                line['y'] = self.np_random.uniform(-line['height'], 0)
                line['x'] = self.np_random.uniform(0, self.WIDTH)

        # 5. Check for termination and calculate reward
        reward = 0
        terminated = False
        
        player_rect = pygame.Rect(
            self.player_x - self.PLAYER_WIDTH / 2, 
            self.PLAYER_Y_POS, 
            self.PLAYER_WIDTH, 
            self.PLAYER_HEIGHT
        )

        if not self.game_over:
            # Check for collision
            for obs_row in self.obstacles:
                rect1 = pygame.Rect(self.TRACK_LEFT, obs_row['y'], obs_row['gap_x'] - self.TRACK_LEFT, self.OBSTACLE_HEIGHT)
                rect2 = pygame.Rect(obs_row['gap_x'] + self.OBSTACLE_GAP_WIDTH, obs_row['y'], self.TRACK_RIGHT - (obs_row['gap_x'] + self.OBSTACLE_GAP_WIDTH), self.OBSTACLE_HEIGHT)
                if player_rect.colliderect(rect1) or player_rect.colliderect(rect2):
                    self.game_over = True
                    terminated = True
                    reward = -100
                    # sfx: explosion
                    for _ in range(50):
                        self.particles.append(Particle(self.player_x, self.PLAYER_Y_POS + self.PLAYER_HEIGHT / 2, self.np_random))
                    break
            
            if not terminated:
                # Check for win condition
                if self.distance_traveled >= self.TOTAL_DISTANCE:
                    self.game_over = True
                    self.win = True
                    terminated = True
                    # sfx: win_jingle
                    time_penalty = 0.01 * self.steps
                    reward = max(0, 50 - time_penalty) # Ensure reward is not negative
                
                # Check for timeout
                elif self.steps >= self.MAX_STEPS:
                    self.game_over = True
                    terminated = True
                    reward = -10
                
                # Continuous reward for survival
                else:
                    reward = 0.1
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_obstacle_row(self, initial_y=None, existing_row=None):
        gap_x = self.np_random.uniform(
            self.TRACK_LEFT, 
            self.TRACK_RIGHT - self.OBSTACLE_GAP_WIDTH
        )
        if existing_row:
            # Repurpose existing row that went off-screen
            max_y = min(o['y'] for o in self.obstacles)
            existing_row['y'] = max_y - self.OBSTACLE_V_SPACING
            existing_row['gap_x'] = gap_x
        else:
            # Create a new row during reset
            self.obstacles.append({'y': initial_y, 'gap_x': gap_x})


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render speed lines (background)
        for line in self.speed_lines:
            start_pos = (int(line['x']), int(line['y']))
            end_pos = (int(line['x']), int(line['y'] + line['height']))
            pygame.draw.line(self.screen, line['color'], start_pos, end_pos, 2)
        
        # Render track lines
        pygame.draw.line(self.screen, self.COLOR_TRACK, (self.TRACK_LEFT, 0), (self.TRACK_LEFT, self.HEIGHT), self.TRACK_LINE_WIDTH)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (self.TRACK_RIGHT, 0), (self.TRACK_RIGHT, self.HEIGHT), self.TRACK_LINE_WIDTH)

        # Render finish line if close
        dist_to_finish = self.TOTAL_DISTANCE - self.distance_traveled
        if dist_to_finish < self.OBSTACLE_V_SPACING * 2:
            finish_y = self.PLAYER_Y_POS - (dist_to_finish / self.scroll_speed) * self.scroll_speed
            if 0 < finish_y < self.HEIGHT:
                pygame.draw.line(self.screen, self.COLOR_FINISH_GLOW, (self.TRACK_LEFT, finish_y), (self.TRACK_RIGHT, finish_y), 10)
                pygame.draw.line(self.screen, self.COLOR_FINISH, (self.TRACK_LEFT, finish_y), (self.TRACK_RIGHT, finish_y), 5)

        # Render obstacles
        for obs in self.obstacles:
            y = int(obs['y'])
            if -self.OBSTACLE_HEIGHT < y < self.HEIGHT:
                # Left part
                rect1 = pygame.Rect(self.TRACK_LEFT, y, obs['gap_x'] - self.TRACK_LEFT, self.OBSTACLE_HEIGHT)
                # Right part
                rect2 = pygame.Rect(obs['gap_x'] + self.OBSTACLE_GAP_WIDTH, y, self.TRACK_RIGHT - (obs['gap_x'] + self.OBSTACLE_GAP_WIDTH), self.OBSTACLE_HEIGHT)
                
                # Draw glow
                glow_rect1 = rect1.inflate(6, 6)
                glow_rect2 = rect2.inflate(6, 6)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect1, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect2, border_radius=3)
                
                # Draw solid rect
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect1, border_radius=2)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect2, border_radius=2)

        # Render particles
        for p in self.particles:
            p.draw(self.screen)

        # Render player
        if not (self.game_over and not self.win): # Hide player on collision
            player_rect_glow = pygame.Rect(
                self.player_x - (self.PLAYER_WIDTH + 6) / 2, 
                self.PLAYER_Y_POS - 3, 
                self.PLAYER_WIDTH + 6, 
                self.PLAYER_HEIGHT + 6
            )
            player_rect = pygame.Rect(
                self.player_x - self.PLAYER_WIDTH / 2, 
                self.PLAYER_Y_POS, 
                self.PLAYER_WIDTH, 
                self.PLAYER_HEIGHT
            )
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, player_rect_glow, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        progress = min(100, int((self.distance_traveled / self.TOTAL_DISTANCE) * 100))
        progress_text = self.font.render(f"PROGRESS: {progress}%", True, self.COLOR_TEXT)
        self.screen.blit(progress_text, (10, 10))
        
        score_text = self.font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            if self.win:
                end_text = self.font.render("FINISH!", True, self.COLOR_FINISH)
            else:
                end_text = self.font.render("GAME OVER", True, self.COLOR_OBSTACLE)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress": self.distance_traveled / self.TOTAL_DISTANCE,
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
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Speeding Line")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human play

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()