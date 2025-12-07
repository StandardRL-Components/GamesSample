
# Generated: 2025-08-27T14:38:42.908412
# Source Brief: brief_00746.md
# Brief Index: 746

        
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
        "Controls: Press space to jump over obstacles to the beat."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated highway, jumping over obstacles to the beat in a side-scrolling rhythm game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto_advance=True, this is the step rate

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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_combo = pygame.font.SysFont("monospace", 32, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game Constants ---
        self.GROUND_Y = self.HEIGHT - 80
        self.GRAVITY = -1.5
        self.JUMP_STRENGTH = 20
        self.PLAYER_X = 100
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 50
        self.MAX_MISSES = 5
        self.MAX_JUMPS = 100
        self.MAX_STEPS = 3000

        # --- Rewards ---
        self.REWARD_SUCCESSFUL_JUMP = 1.0
        self.REWARD_SAFE_JUMP = -0.2
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -50.0

        # --- Colors ---
        self.COLOR_BG = (16, 0, 32)
        self.COLOR_ROAD = (40, 20, 60)
        self.COLOR_LINES = (0, 255, 255)
        self.COLOR_PLAYER = (255, 255, 0)
        self.OBSTACLE_COLORS = [(255, 65, 54), (0, 116, 217), (57, 204, 204), (61, 153, 112)]
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_FLASH = (255, 0, 0)

        # --- State Variables ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_y = None
        self.player_vy = None
        self.is_jumping = None
        self.misses = None
        self.combo = None
        self.jumps_succeeded = None
        self.obstacles = None
        self.particles = None
        self.obstacle_track = None
        self.track_idx = None
        self.initial_obstacle_speed = None
        self.obstacle_speed = None
        self.combo_reward_milestone = None
        self.flash_timer = None
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional validation call
    
    def _generate_track(self):
        """Creates a procedural list of obstacles to be spawned."""
        self.obstacle_track = []
        current_step = 90  # Start first obstacle after 3 seconds
        for _ in range(self.MAX_JUMPS):
            spawn_step = current_step
            height = self.np_random.integers(30, 81)
            width = self.np_random.integers(20, 41)
            color = random.choice(self.OBSTACLE_COLORS)
            self.obstacle_track.append({
                "spawn_step": spawn_step,
                "height": height,
                "width": width,
                "color": color,
            })
            # Time between obstacles: 1s, 1.5s, or 2s
            current_step += self.np_random.choice([30, 45, 60])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.is_jumping = False
        
        self.misses = 0
        self.combo = 0
        self.jumps_succeeded = 0
        
        self.obstacles = []
        self.particles = []
        
        self._generate_track()
        self.track_idx = 0
        
        self.initial_obstacle_speed = 6.0
        self.obstacle_speed = self.initial_obstacle_speed
        
        self.combo_reward_milestone = 10
        self.flash_timer = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right (unused)
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        if space_held and not self.is_jumping:
            self.is_jumping = True
            self.player_vy = self.JUMP_STRENGTH
            # Sound: Jump sfx
            
            is_safe = True
            for obs in self.obstacles:
                if self.PLAYER_X < obs['x'] < self.PLAYER_X + 300:
                    is_safe = False
                    break
            if is_safe:
                reward += self.REWARD_SAFE_JUMP

        # Update game logic
        self.steps += 1
        if self.flash_timer > 0:
            self.flash_timer -= 1

        if self.is_jumping:
            self.player_vy += self.GRAVITY
            self.player_y -= self.player_vy
        
        if self.player_y > self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            self.is_jumping = False

        if self.track_idx < len(self.obstacle_track) and self.steps >= self.obstacle_track[self.track_idx]["spawn_step"]:
            obs_data = self.obstacle_track[self.track_idx]
            self.obstacles.append({
                "x": self.WIDTH,
                "y": self.GROUND_Y - obs_data["height"],
                "width": obs_data["width"],
                "height": obs_data["height"],
                "color": obs_data["color"],
                "cleared": False,
            })
            self.track_idx += 1

        for obs in self.obstacles:
            old_x = obs['x']
            obs['x'] -= self.obstacle_speed
            
            if not obs['cleared'] and old_x >= self.PLAYER_X and obs['x'] < self.PLAYER_X:
                player_rect = pygame.Rect(self.PLAYER_X, self.player_y - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
                obs_rect = pygame.Rect(obs['x'], obs['y'], obs['width'], obs['height'])
                
                if player_rect.colliderect(obs_rect):
                    self.misses += 1
                    self.combo = 0
                    self.combo_reward_milestone = 10
                    self.flash_timer = 5
                    # Sound: Fail/Hit sfx
                else:
                    self.jumps_succeeded += 1
                    self.combo += 1
                    reward += self.REWARD_SUCCESSFUL_JUMP
                    self._create_particles(self.PLAYER_X + self.PLAYER_WIDTH / 2, self.GROUND_Y, obs['color'])
                    # Sound: Success sfx
                    
                    if self.combo >= self.combo_reward_milestone:
                        reward += self.combo_reward_milestone / 2.0
                        self.combo_reward_milestone += 10
                
                obs['cleared'] = True
        
        self.obstacles = [obs for obs in self.obstacles if obs['x'] + obs['width'] > 0]
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1
            p['life'] -= 1

        if self.steps > 0 and self.steps % 50 == 0:
            self.obstacle_speed += 0.05
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.misses >= self.MAX_MISSES:
                reward += self.REWARD_LOSE
            elif self.jumps_succeeded >= self.MAX_JUMPS:
                reward += self.REWARD_WIN
        
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self):
        if self.misses >= self.MAX_MISSES:
            return True
        if self.jumps_succeeded >= self.MAX_JUMPS:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _create_particles(self, x, y, color):
        for _ in range(20):
            self.particles.append({
                'x': x, 'y': y,
                'vx': self.np_random.uniform(-3, 3), 'vy': self.np_random.uniform(-5, -1),
                'life': self.np_random.integers(15, 30), 'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "misses": self.misses,
            "jumps_succeeded": self.jumps_succeeded
        }

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_ROAD, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        horizon_y = self.GROUND_Y - 150
        vanishing_point = (self.WIDTH // 2, horizon_y + 50)
        
        for i in range(1, 10):
            p1_y = self.GROUND_Y + (i * i)
            if p1_y > self.HEIGHT: break
            pygame.draw.line(self.screen, self.COLOR_LINES, (0, p1_y), (self.WIDTH, p1_y), 1)

        for i in range(1, 15):
            x = self.WIDTH/2 + (i * i * 1.5); pygame.draw.line(self.screen, self.COLOR_LINES, (x, self.GROUND_Y), vanishing_point)
            x = self.WIDTH/2 - (i * i * 1.5); pygame.draw.line(self.screen, self.COLOR_LINES, (x, self.GROUND_Y), vanishing_point)
        pygame.draw.line(self.screen, self.COLOR_LINES, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)
        
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, obs['color'], (int(obs['x']), int(obs['y']), int(obs['width']), int(obs['height'])))
            highlight_color = tuple(min(255, c + 50) for c in obs['color'])
            pygame.draw.line(self.screen, highlight_color, (int(obs['x']), int(obs['y'])), (int(obs['x'] + obs['width']), int(obs['y'])), 2)

        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), (*p['color'], alpha))
        
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        glow_rect = player_rect.inflate(10, 10)
        pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_PLAYER, 50))

    def _render_ui(self):
        for i in range(self.MAX_MISSES):
            color = self.COLOR_FLASH if i < self.misses else (100, 100, 100)
            pygame.draw.circle(self.screen, color, (self.WIDTH - 30 - i * 25, 30), 8)

        if self.combo > 1:
            font_size = min(120, 32 + self.combo)
            dynamic_font = pygame.font.SysFont("monospace", font_size, bold=True)
            text_combo = dynamic_font.render(f"{self.combo}x", True, self.COLOR_TEXT)
            self.screen.blit(text_combo, text_combo.get_rect(topleft=(20, 20)))
        
        text_jumps = self.font_ui.render(f"Jumps: {self.jumps_succeeded}/{self.MAX_JUMPS}", True, self.COLOR_TEXT)
        self.screen.blit(text_jumps, (20, self.HEIGHT - 40))

        if self.flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.flash_timer / 5))
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
            msg = "LEVEL COMPLETE" if self.jumps_succeeded >= self.MAX_JUMPS else "GAME OVER"
            color = self.COLOR_LINES if self.jumps_succeeded >= self.MAX_JUMPS else self.COLOR_FLASH
            text_surface = self.font_game_over.render(msg, True, color)
            self.screen.blit(text_surface, text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(reward, (int, float)) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        
        self.reset()
        initial_speed = self.obstacle_speed
        for i in range(1, 101):
            self.step(self.action_space.sample())
            if i % 50 == 0:
                expected_speed = initial_speed + 0.05 * (i // 50)
                assert abs(self.obstacle_speed - expected_speed) < 1e-9, f"Speed mismatch at step {i}"

        self.reset(); self.misses = self.MAX_MISSES - 1; _, _, term, _, _ = self.step(self.action_space.sample()); assert term, "Game should terminate after MAX_MISSES"
        self.reset(); self.jumps_succeeded = self.MAX_JUMPS; _, reward, term, _, _ = self.step(self.action_space.sample()); assert term and reward >= self.REWARD_WIN, "Win condition fail"
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Jumper")
    clock = pygame.time.Clock()
    
    running = True
    done = False
    total_reward = 0.0
    
    while running:
        space_pressed = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    done = False
                if event.key == pygame.K_q:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_pressed = 1
            
        action = [0, space_pressed, 0]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        
        font = pygame.font.SysFont("monospace", 16)
        score_text = font.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
        render_screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()