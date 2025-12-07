
# Generated: 2025-08-27T20:53:37.013250
# Source Brief: brief_02610.md
# Brief Index: 2610

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to run, ↑ or Space to jump. Hold Shift to dash."
    )

    game_description = (
        "Control a robot in a fast-paced, neon-drenched obstacle course. Jump and dash to avoid barriers, "
        "collect coins, and race to the finish line before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 60
        self.LEVEL_LENGTH = 8000  # pixels
        self.GAME_DURATION_SECONDS = 30

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GRID = (30, 30, 60)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 150)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (150, 25, 25)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_FINISH = (50, 255, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_DASH_TRAIL = (100, 220, 255)
        self.COLOR_DASH_UI_READY = (100, 255, 255)
        self.COLOR_DASH_UI_COOLDOWN = (60, 100, 100)

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 28, bold=True)
        
        # Game Physics & Mechanics
        self.GRAVITY = 0.4
        self.PLAYER_X_SPEED = 3.5
        self.JUMP_STRENGTH = -10.5
        self.GROUND_Y = self.SCREEN_HEIGHT - 60
        self.BASE_SCROLL_SPEED = 4.0
        self.DASH_SPEED_BOOST = 8
        self.DASH_DURATION = 15  # frames
        self.DASH_COOLDOWN = 90  # frames
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = None
        self.is_dashing = None
        self.dash_timer = None
        self.dash_cooldown_timer = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_timer = None
        self.world_scroll = None
        self.scroll_speed = None
        self.obstacles = None
        self.coins = None
        self.particles = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(100, self.GROUND_Y)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = True
        
        self.is_dashing = False
        self.dash_timer = 0
        self.dash_cooldown_timer = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = self.GAME_DURATION_SECONDS * self.FPS
        self.world_scroll = 0
        self.scroll_speed = self.BASE_SCROLL_SPEED

        self.obstacles = []
        self.coins = []
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        current_x = 800
        while current_x < self.LEVEL_LENGTH - 1000:
            gap_size = self.np_random.integers(350, 600)
            current_x += gap_size
            
            obstacle_height = self.np_random.integers(40, 120)
            obstacle_y = self.GROUND_Y - obstacle_height + 20
            obstacle_width = self.np_random.integers(30, 60)
            self.obstacles.append({
                "rect": pygame.Rect(current_x, obstacle_y, obstacle_width, obstacle_height),
                "rewarded_dash": False
            })
            
            num_coins = self.np_random.integers(1, 4)
            for i in range(num_coins):
                coin_x = current_x - gap_size + 150 + i * 70
                coin_y_offset = self.np_random.integers(-80, 20)
                self.coins.append(pygame.Rect(coin_x, self.GROUND_Y - 50 + coin_y_offset, 15, 15))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01  # Survival reward
        
        self._handle_input(action)
        self._update_player_state()
        self._update_world_state()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self.steps += 1
        self.game_timer -= 1
        
        if self.steps > 0 and self.steps % 500 == 0:
            self.scroll_speed = min(self.scroll_speed + 0.5, self.BASE_SCROLL_SPEED * 2)

        terminated = self.game_over
        win_condition_met = self.world_scroll >= self.LEVEL_LENGTH

        if win_condition_met:
            terminated = True
            self.game_over = True
            reward += 100  # Reached finish line
        elif self.game_timer <= 0:
            terminated = True
            self.game_over = True
            reward += -50  # Time ran out
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.player_vel.x = 0
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_X_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_X_SPEED
            
        if (movement == 1 or space_held) and self.is_grounded:
            self.player_vel.y = self.JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump
            self._spawn_particles(self.player_pos + pygame.Vector2(0, 20), 10, self.COLOR_PLAYER, 10, (1, 3))

        if shift_held and self.dash_cooldown_timer == 0 and not self.is_dashing:
            self.is_dashing = True
            self.dash_timer = self.DASH_DURATION
            self.dash_cooldown_timer = self.DASH_COOLDOWN
            # sfx: dash

    def _update_player_state(self):
        if self.is_dashing:
            self.dash_timer -= 1
            if self.dash_timer <= 0:
                self.is_dashing = False
            else:
                if self.steps % 2 == 0:
                    self._spawn_particles(self.player_pos, 5, self.COLOR_DASH_TRAIL, 3, (0.5, 1), pygame.Vector2(-1, 0))
        
        if self.dash_cooldown_timer > 0:
            self.dash_cooldown_timer -= 1

        if not self.is_grounded:
            self.player_vel.y += self.GRAVITY
        
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y
        
        if self.player_pos.y >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel.y = 0
            if not self.is_grounded: # Landed
                # sfx: land
                self._spawn_particles(self.player_pos + pygame.Vector2(0, 20), 5, self.COLOR_PLAYER, 5, (0.5, 2))
            self.is_grounded = True
            
        self.player_pos.x = max(20, min(self.player_pos.x, self.SCREEN_WIDTH - 20))

    def _update_world_state(self):
        current_scroll = self.scroll_speed
        if self.is_dashing:
            current_scroll += self.DASH_SPEED_BOOST
        self.world_scroll += current_scroll
        
        for obs_data in self.obstacles:
            obs_data["rect"].x -= current_scroll
        for coin in self.coins:
            coin.x -= current_scroll
            
        self.obstacles = [o for o in self.obstacles if o["rect"].right > 0]
        self.coins = [c for c in self.coins if c.right > 0]
        
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['size'] -= 0.25
            if p['size'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 40)
        
        for obs_data in self.obstacles:
            if player_rect.colliderect(obs_data["rect"]):
                if not self.is_dashing:
                    self.game_over = True
                    # sfx: explosion
                    self._spawn_particles(self.player_pos, 10, self.COLOR_OBSTACLE, 50, (2, 6))
                    return -100
                elif not obs_data["rewarded_dash"]:
                    obs_data["rewarded_dash"] = True
                    reward += 5
                    # sfx: dash_through
                    self._spawn_particles(pygame.Vector2(obs_data["rect"].center), 12, self.COLOR_OBSTACLE, 20, (3, 5))
        
        for coin in self.coins[:]:
            if player_rect.colliderect(coin):
                self.coins.remove(coin)
                self.score += 1
                reward += 1
                # sfx: coin_pickup
                self._spawn_particles(pygame.Vector2(coin.center), 8, self.COLOR_COIN, 15, (1, 4))
                
        return reward

    def _spawn_particles(self, pos, size, color, count, speed_range, direction=None):
        for _ in range(count):
            if direction is None:
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel_vec = pygame.Vector2(math.cos(angle), math.sin(angle))
            else:
                angle = math.atan2(direction.y, direction.x) + self.np_random.uniform(-0.5, 0.5)
                vel_vec = pygame.Vector2(math.cos(angle), math.sin(angle))

            speed = self.np_random.uniform(speed_range[0], speed_range[1])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel_vec * speed,
                'size': self.np_random.uniform(size * 0.5, size),
                'color': color
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Parallax background grid
        parallax_scroll = self.world_scroll * 0.5
        for i in range(0, self.SCREEN_WIDTH + 100, 50):
            x = i - (parallax_scroll % 50)
            pygame.draw.line(self.screen, self.COLOR_GRID, (int(x), 0), (int(x), self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

        # Ground
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.GROUND_Y + 20), (self.SCREEN_WIDTH, self.GROUND_Y + 20), 3)

        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), p['color'])

        # Finish Line
        finish_x = self.LEVEL_LENGTH - self.world_scroll
        if finish_x < self.SCREEN_WIDTH + 50:
            pygame.draw.rect(self.screen, self.COLOR_FINISH, (finish_x, 0, 10, self.SCREEN_HEIGHT))

        # Obstacles
        for obs_data in self.obstacles:
            obs = obs_data["rect"]
            glow_rect = obs.inflate(10, 10)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, border_radius=3)

        # Coins
        for coin in self.coins:
            bob = math.sin(self.steps * 0.1 + coin.x * 0.05) * 3
            spin = (math.cos(self.steps * 0.2 + coin.y * 0.05) * 0.4 + 0.6)
            pos = (int(coin.centerx), int(coin.centery + bob))
            pygame.gfxdraw.filled_ellipse(self.screen, pos[0], pos[1], int(coin.width * spin / 2), int(coin.height / 2), self.COLOR_COIN)

        # Player
        if not (self.game_over and collision_reward < -50): # Hide player on death
            player_rect = pygame.Rect(0, 0, 20, 40)
            player_rect.center = (int(self.player_pos.x), int(self.player_pos.y - 10))

            # Glow
            glow_surface = pygame.Surface((40, 60), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surface, self.COLOR_PLAYER_GLOW + (150,), (0, 0, 40, 60))
            self.screen.blit(glow_surface, (player_rect.centerx - 20, player_rect.centery - 30))

            # Body
            if self.is_dashing:
                dash_rect = player_rect.copy()
                dash_rect.w += 20
                dash_rect.centerx -= 10
                pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, dash_rect)
            else:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)

            # Simple leg animation
            if self.is_grounded and abs(self.player_vel.x) > 0.1:
                leg_offset = math.sin(self.steps * 0.5) * 5
                leg_rect = pygame.Rect(player_rect.centerx - 5 + leg_offset, player_rect.bottom - 5, 10, 5)
                pygame.draw.rect(self.screen, self.COLOR_BG, leg_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Timer
        time_left = max(0, self.game_timer / self.FPS)
        secs = int(time_left % 60)
        timer_text = self.font_timer.render(f"{secs:02}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - 60, 10))
        
        # Dash Cooldown
        dash_label = self.font_ui.render("DASH", True, self.COLOR_TEXT)
        self.screen.blit(dash_label, (20, 45))
        bar_w = 100
        bar_h = 10
        
        # Cooldown progress
        cooldown_progress = self.dash_cooldown_timer / self.DASH_COOLDOWN
        fill_w = bar_w * (1 - cooldown_progress)
        
        color = self.COLOR_DASH_UI_READY if self.dash_cooldown_timer == 0 else self.COLOR_DASH_UI_COOLDOWN
        pygame.draw.rect(self.screen, self.COLOR_GRID, (80, 47, bar_w, bar_h))
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (80, 47, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (80, 47, bar_w, bar_h), 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": round(max(0, self.game_timer / self.FPS), 2),
            "progress": round(self.world_scroll / self.LEVEL_LENGTH, 2)
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use Pygame for human interaction
    pygame.display.set_caption("RoboRun")
    display_screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    while not terminated:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            terminated = False

        clock.tick(env.FPS)
        
    env.close()