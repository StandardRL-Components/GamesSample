
# Generated: 2025-08-28T05:58:48.232423
# Source Brief: brief_02779.md
# Brief Index: 2779

        
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

    # User-facing control string, corrected to match the implemented side-scroller mechanics
    user_guide = (
        "Controls: Use ↑/↓ to move vertically. Use → to accelerate and ← to brake. Press Space to boost."
    )

    # User-facing description of the game
    game_description = (
        "A fast-paced, retro-styled side-scrolling racer. Dodge obstacles, manage your speed, and boost to reach the finish line against the clock."
    )

    # Frames auto-advance at a fixed rate for smooth real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # The brief asks for 30fps for auto-advance
        self.MAX_STEPS = 1800 # 60 seconds at 30 FPS

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_TRACK = (60, 60, 80)
        self.COLOR_TRACK_LINES = (90, 90, 110)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 150, 150)
        self.COLOR_OBSTACLE = (120, 120, 140)
        self.COLOR_BOOST_PARTICLE = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_FINISH_A = (255, 255, 255)
        self.COLOR_FINISH_B = (30, 30, 30)

        # Player Physics
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 30, 15
        self.PLAYER_Y_SPEED = 6.0
        self.ACCELERATION = 0.2
        self.BRAKING = 0.4
        self.FRICTION = 0.05
        self.MAX_SPEED = 15.0
        self.MIN_SPEED = 1.0
        self.BOOST_SPEED_BONUS = 15.0
        self.BOOST_DURATION = 15 # frames

        # Track & World
        self.TRACK_Y_TOP = 100
        self.TRACK_Y_BOTTOM = self.HEIGHT - 100
        self.FINISH_LINE_DISTANCE = 20000 # Total distance to travel

        # Obstacles
        self.OBSTACLE_MIN_W, self.OBSTACLE_MAX_W = 20, 60
        self.OBSTACLE_MIN_H, self.OBSTACLE_MAX_H = 20, 80
        self.INITIAL_OBSTACLE_SPAWN_RATE = 45 # frames between spawns
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 50)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_speed = None
        self.world_progress = None
        self.obstacles = None
        self.particles = None
        self.boost_timer = None
        self.obstacle_spawn_timer = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.pending_boost_rewards = None
        self.obstacle_spawn_rate = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(100, self.HEIGHT / 2)
        self.player_speed = self.MIN_SPEED
        self.world_progress = 0.0

        self.obstacles = []
        self.particles = []
        
        self.boost_timer = 0
        self.obstacle_spawn_rate = self.INITIAL_OBSTACLE_SPAWN_RATE
        self.obstacle_spawn_timer = self.obstacle_spawn_rate

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        # For delayed boost reward
        self.pending_boost_rewards = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1 # Survival reward

        # --- 1. Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Vertical Movement
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_Y_SPEED
        elif movement == 2: # Down
            self.player_pos.y += self.PLAYER_Y_SPEED
        
        # Horizontal Speed Control
        if movement == 4: # Accelerate
            self.player_speed += self.ACCELERATION
        elif movement == 3: # Brake
            self.player_speed -= self.BRAKING
            reward -= 0.1 # Small penalty for braking
        else: # Coast (friction)
            self.player_speed -= self.FRICTION

        # Boost
        if space_held and self.boost_timer == 0:
            self.boost_timer = self.BOOST_DURATION
            # Add a pending reward to be resolved in 10 frames
            self.pending_boost_rewards.append({'resolve_step': self.steps + 10})
            # Sound effect placeholder: # sfx_boost_start()

        # --- 2. Update Game State ---
        # Apply boost if active
        effective_speed = self.player_speed
        if self.boost_timer > 0:
            self.boost_timer -= 1
            effective_speed += self.BOOST_SPEED_BONUS
            # Spawn boost particles
            if self.steps % 2 == 0:
                self._spawn_particle(self.player_pos.x, self.player_pos.y + self.PLAYER_HEIGHT / 2)
        
        # Clamp speed and player position
        self.player_speed = max(self.MIN_SPEED, min(self.player_speed, self.MAX_SPEED))
        self.player_pos.y = max(self.TRACK_Y_TOP, min(self.player_pos.y, self.TRACK_Y_BOTTOM - self.PLAYER_HEIGHT))
        
        # Update world progress
        self.world_progress += effective_speed
        
        # Update obstacles
        for obs in self.obstacles:
            obs['rect'].x -= effective_speed
            # Check for successful dodge
            if not obs['dodged'] and obs['rect'].right < self.player_pos.x:
                obs['dodged'] = True
                reward += 5.0
                # Sound effect placeholder: # sfx_dodge()

        # Update particles
        for p in self.particles:
            p['pos'].x -= self.np_random.uniform(0.8, 1.2) * effective_speed
            p['pos'].y += p['vel'].y
            p['lifetime'] -= 1
            p['radius'] -= 0.2

        # Clean up off-screen/dead entities
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['radius'] > 0]
        
        # Spawn new obstacles
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()
            self.obstacle_spawn_timer = self.obstacle_spawn_rate

        # --- 3. Check for Termination & Rewards ---
        terminated = False
        
        # Resolve pending boost rewards
        new_pending_rewards = []
        for pbr in self.pending_boost_rewards:
            if pbr['resolve_step'] == self.steps:
                reward += 10.0 # Successful boost
            else:
                new_pending_rewards.append(pbr)
        self.pending_boost_rewards = new_pending_rewards

        # Check for win condition
        if self.world_progress >= self.FINISH_LINE_DISTANCE:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward = 100.0
            # Sound effect placeholder: # sfx_win_jingle()
        
        # Check for collisions
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                self.game_over = True
                terminated = True
                reward = -100.0
                # Penalize reckless boosting
                if self.boost_timer > 0:
                    reward -= 50.0
                # Clear pending boost rewards on crash
                self.pending_boost_rewards = []
                # Sound effect placeholder: # sfx_crash_explosion()
                break
        
        # Check for timeout
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True
            reward = -50.0 # Penalty for running out of time

        self.score += reward
        
        # --- 4. Return Gym Tuple ---
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_particle(self, x, y):
        self.particles.append({
            'pos': pygame.Vector2(x, y),
            'vel': pygame.Vector2(0, self.np_random.uniform(-1, 1)),
            'radius': self.np_random.uniform(3, 6),
            'lifetime': 20
        })

    def _spawn_obstacle(self):
        h = self.np_random.integers(self.OBSTACLE_MIN_H, self.OBSTACLE_MAX_H + 1)
        w = self.np_random.integers(self.OBSTACLE_MIN_W, self.OBSTACLE_MAX_W + 1)
        y = self.np_random.integers(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM - h + 1)
        self.obstacles.append({
            'rect': pygame.Rect(self.WIDTH + w, y, w, h),
            'dodged': False
        })
        
    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "speed": self.player_speed,
            "progress": self.world_progress / self.FINISH_LINE_DISTANCE
        }

    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)

        # Track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP, self.WIDTH, self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP))
        pygame.draw.line(self.screen, self.COLOR_TRACK_LINES, (0, self.TRACK_Y_TOP), (self.WIDTH, self.TRACK_Y_TOP), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK_LINES, (0, self.TRACK_Y_BOTTOM), (self.WIDTH, self.TRACK_Y_BOTTOM), 2)

        # Scrolling road lines
        line_y_positions = [self.TRACK_Y_TOP + (self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP) * p for p in [0.25, 0.5, 0.75]]
        for y in line_y_positions:
            offset = (self.world_progress * 0.5) % 80
            for i in range(self.WIDTH // 80 + 2):
                start_x = i * 80 - offset
                pygame.draw.line(self.screen, self.COLOR_TRACK_LINES, (start_x, y), (start_x + 40, y), 1)

        # Particles (drawn before player)
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 20))))
            color = (*self.COLOR_BOOST_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Player
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        # Glow effect
        glow_rect = player_rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        # Car body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        
        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'], border_radius=2)

        # Finish Line
        finish_x = self.WIDTH - (self.world_progress - self.FINISH_LINE_DISTANCE)
        if finish_x < self.WIDTH * 1.5: # Only draw when it's getting close
            check_size = 20
            for y in range(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM, check_size):
                for x_offset in range(0, 40, check_size * 2): # Draw two columns for thickness
                    color = self.COLOR_FINISH_A if (y // check_size) % 2 == 0 else self.COLOR_FINISH_B
                    pygame.draw.rect(self.screen, color, (finish_x + x_offset, y, check_size, check_size))
                    color = self.COLOR_FINISH_B if (y // check_size) % 2 == 0 else self.COLOR_FINISH_A
                    pygame.draw.rect(self.screen, color, (finish_x + x_offset + check_size, y, check_size, check_size))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Speed
        speed_kmh = self.player_speed * 10
        speed_text = self.font_ui.render(f"SPEED: {int(speed_kmh):03d} KM/H", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.WIDTH - speed_text.get_width() - 10, 10))
        
        # Time
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))

        # Progress Bar
        progress_pct = min(1.0, self.world_progress / self.FINISH_LINE_DISTANCE)
        bar_width = self.WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (10, self.HEIGHT - 25, bar_width, 15))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.HEIGHT - 25, bar_width * progress_pct, 15))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, self.HEIGHT - 25, bar_width, 15), 1)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "FINISH!" if self.game_won else "CRASHED"
            if not self.game_won and self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
            
            end_text = self.font_game_over.render(msg, True, self.COLOR_PLAYER if self.game_won else self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    
    terminated = False
    total_reward = 0
    
    # --- Main Game Loop ---
    while not terminated:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        
        # Movement
        movement_action = 0 # no-op
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
            
        # Buttons
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # Pygame uses a different coordinate system for surfaces, so we need to transpose
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    
    # Keep the final screen visible for a moment
    pygame.time.wait(2000)
    
    env.close()