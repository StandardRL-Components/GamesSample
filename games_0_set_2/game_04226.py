import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
from collections import deque
import os
import pygame


# Set the video driver to dummy to run headless, required for server-side evaluation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Press Space to jump. Time your jumps with the beat to score more points."
    )

    # Short, user-facing description of the game
    game_description = (
        "A minimalist rhythm-action game. Jump over obstacles in sync with the beat to achieve a flow state and maximize your score. Vibrant particle effects reward perfect timing."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (20, 10, 40)
    COLOR_BG_BOTTOM = (60, 30, 80)
    COLOR_GROUND = (80, 60, 100)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_OBSTACLE = (210, 210, 220)
    COLOR_BEAT_INDICATOR = (255, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_MISS = (255, 50, 50)

    # Player
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 30
    PLAYER_START_X = 100
    PLAYER_GROUND_Y = 320
    JUMP_VELOCITY = -13
    GRAVITY = 0.6

    # Game
    MAX_STEPS = 2000
    MAX_MISSES = 3
    GROUND_HEIGHT = 80
    
    # Rhythm
    BEAT_PERIOD = 40  # steps per beat (30fps -> 1.33s per beat)
    PERFECT_TIMING_WINDOW = 3 # steps before/after beat for 'perfect' bonus

    # Obstacles
    OBSTACLE_WIDTH = 25
    OBSTACLE_HEIGHT = 40
    OBSTACLE_MIN_GAP = 150
    OBSTACLE_MAX_GAP = 300
    INITIAL_OBSTACLE_SPEED = 4.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        self._prepare_background()
        
        self.player_pos = None
        self.player_vel_y = None
        self.is_jumping = None
        self.jump_initiated_on_beat = None
        
        self.obstacles = None
        self.obstacle_speed = None
        self.next_obstacle_x = None

        self.particles = None
        
        self.steps = None
        self.score = None
        self.misses = None
        self.game_over = None
        self.beat_timer = None
        
    def _prepare_background(self):
        """Pre-renders the background gradient for performance."""
        self.bg_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.SCREEN_WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.PLAYER_START_X, self.PLAYER_GROUND_Y]
        self.player_vel_y = 0
        self.is_jumping = False
        self.jump_initiated_on_beat = False
        
        self.obstacles = deque()
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.next_obstacle_x = self.SCREEN_WIDTH + 100

        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.beat_timer = 0
        
        self._spawn_initial_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        # The original code had self.game_over check here, but Gymnasium's API
        # expects step to be callable even after termination. The correct behavior
        # is to return the final state without updating the environment.
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if space_held and not self.is_jumping:
            # Sound: Jump
            self.is_jumping = True
            self.player_vel_y = self.JUMP_VELOCITY
            self.jump_initiated_on_beat = self._is_on_beat()
    
        # --- Game Logic Update ---
        self.steps += 1
        self.beat_timer = (self.beat_timer + 1) % self.BEAT_PERIOD
        reward = 0.1  # Survival reward

        # Update player
        self._update_player()
        
        # Update obstacles
        self._update_obstacles()
        
        # Update particles
        self._update_particles()
        
        # Update difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed += 0.2

        # --- Collision and Reward ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'], obs['y'], self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT)
            
            # Collision
            if not obs['collided'] and player_rect.colliderect(obs_rect):
                # Sound: Miss / Hit
                self.misses += 1
                reward = -1.0
                obs['collided'] = True
                self._create_particles(player_rect.center, 30, self.COLOR_MISS)
                if self.misses >= self.MAX_MISSES:
                    self.game_over = True
                break # Only one collision per frame

            # Successful clear
            if not obs['cleared'] and player_rect.right > obs_rect.right and self.is_jumping:
                obs['cleared'] = True
                if self.jump_initiated_on_beat:
                    # Sound: Perfect Clear
                    reward += 1.0
                    self.score += 100
                    # Perfect timing particles
                    self._create_particles(player_rect.center, 50, self.COLOR_PLAYER, life=40, spread=90, multi_color=True)
                else:
                    # Sound: Good Clear
                    reward += 0.5
                    self.score += 50
                    # Good timing particles
                    self._create_particles(player_rect.center, 25, self.COLOR_PLAYER, life=30, spread=60)

        # --- Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Reached the end
            reward += 100
            self.score += 1000
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self):
        if self.is_jumping:
            self.player_vel_y += self.GRAVITY
            self.player_pos[1] += self.player_vel_y
            
            if self.player_pos[1] >= self.PLAYER_GROUND_Y:
                self.player_pos[1] = self.PLAYER_GROUND_Y
                self.player_vel_y = 0
                self.is_jumping = False
                # Sound: Land
    
    def _update_obstacles(self):
        # Move all obstacles and the reference point for the next spawn
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed
        self.next_obstacle_x -= self.obstacle_speed
        
        # Remove obstacles that have gone off-screen
        if self.obstacles and self.obstacles[0]['x'] < -self.OBSTACLE_WIDTH:
            self.obstacles.popleft()
        
        # Conditionally spawn a new obstacle if the spawn point gets close to the screen
        if self.next_obstacle_x - self.SCREEN_WIDTH < self.obstacle_speed * 2:
            self._spawn_obstacle()

    def _spawn_initial_obstacles(self):
        while self.next_obstacle_x < self.SCREEN_WIDTH * 2:
            self._spawn_obstacle()

    def _spawn_obstacle(self):
        """Spawns a single obstacle and updates the next spawn position."""
        gap = self.np_random.integers(self.OBSTACLE_MIN_GAP, self.OBSTACLE_MAX_GAP)
        self.obstacles.append({
            'x': self.next_obstacle_x,
            'y': self.PLAYER_GROUND_Y + self.PLAYER_HEIGHT - self.OBSTACLE_HEIGHT,
            'cleared': False,
            'collided': False
        })
        self.next_obstacle_x += self.OBSTACLE_WIDTH + gap

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

    def _create_particles(self, pos, count, color, life=20, speed=3, spread=360, multi_color=False):
        for _ in range(count):
            angle = math.radians(self.np_random.uniform(0, spread) - spread / 2)
            p_speed = self.np_random.uniform(speed * 0.5, speed)
            p_color = color
            if multi_color:
                p_color = (
                    self.np_random.integers(0, 100),
                    self.np_random.integers(150, 255),
                    255
                )
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * p_speed, 'vy': math.sin(angle) * p_speed,
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(life // 2, life),
                'color': p_color
            })

    def _is_on_beat(self):
        return self.beat_timer <= self.PERFECT_TIMING_WINDOW or \
               self.beat_timer >= self.BEAT_PERIOD - self.PERFECT_TIMING_WINDOW

    def _get_observation(self):
        # --- Background ---
        self.screen.blit(self.bg_surface, (0, 0))
        
        # --- Ground ---
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT, self.SCREEN_WIDTH, self.GROUND_HEIGHT))
        
        # --- Beat Indicator ---
        beat_progress = self.beat_timer / self.BEAT_PERIOD
        pulse = (math.sin(beat_progress * 2 * math.pi - math.pi/2) + 1) / 2 # 0 -> 1 -> 0
        pulse = 1 - (1 - pulse)**3 # Ease-out cubic for a sharper pulse
        
        indicator_alpha = 50 + 150 * pulse
        indicator_radius = int(20 + 20 * pulse)
        indicator_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - self.GROUND_HEIGHT // 2)
        
        # Use gfxdraw for antialiasing, drawing to a temporary surface for alpha blending
        temp_surface = pygame.Surface((indicator_radius * 2, indicator_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surface, indicator_radius, indicator_radius, indicator_radius, (*self.COLOR_BEAT_INDICATOR, int(indicator_alpha)))
        pygame.gfxdraw.aacircle(temp_surface, indicator_radius, indicator_radius, indicator_radius, (*self.COLOR_BEAT_INDICATOR, int(indicator_alpha)))
        self.screen.blit(temp_surface, (indicator_pos[0] - indicator_radius, indicator_pos[1] - indicator_radius))
        
        # --- Obstacles ---
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (int(obs['x']), int(obs['y']), self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT))
            if obs['collided']:
                pygame.draw.rect(self.screen, self.COLOR_MISS, (int(obs['x']), int(obs['y']), self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT), 3)

        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20)) if p['life'] < 20 else 255
            color_with_alpha = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), color_with_alpha)
            except (TypeError, ValueError): # Catch potential errors if color or radius are invalid
                pass

        # --- Player ---
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        # Glow effect
        glow_radius = int(self.PLAYER_WIDTH * 0.8)
        glow_alpha = 100
        temp_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surface, glow_radius, glow_radius, glow_radius, (*self.COLOR_PLAYER_GLOW, glow_alpha))
        pygame.gfxdraw.aacircle(temp_surface, glow_radius, glow_radius, glow_radius, (*self.COLOR_PLAYER_GLOW, glow_alpha))
        self.screen.blit(temp_surface, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # --- UI ---
        score_text = self.font_large.render(f"{self.score:06}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        miss_text = self.font_small.render("MISSES", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - 90, 10))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_MISS if i < self.misses else (100, 100, 100)
            pygame.draw.circle(self.screen, color, (self.SCREEN_WIDTH - 75 + i * 25, 40), 8)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "obstacle_speed": self.obstacle_speed,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # To run this __main__ block for local testing, you might need to comment out:
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    # The dummy driver is required for headless operation but prevents window creation.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythm Jumper")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Human Input ---
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # --- Rendering ---
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose the observation back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Control human play speed

    env.close()