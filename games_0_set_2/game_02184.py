
# Generated: 2025-08-27T19:33:18.612429
# Source Brief: brief_02184.md
# Brief Index: 2184

        
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
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Press Space to jump over obstacles."

    # Must be a short, user-facing description of the game:
    game_description = "A minimalist side-view racing game where the player uses a single button to jump over procedurally generated obstacles and reach the finish line."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_TRACK = (180, 180, 190)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 150, 150, 50)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_FINISH_LINE = (50, 200, 50)
    COLOR_PARTICLE_JUMP = (255, 255, 220)
    COLOR_PARTICLE_CRASH = (255, 100, 50)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_UI_SUCCESS = (100, 255, 100)
    COLOR_UI_FAIL = (255, 100, 100)

    # Player
    PLAYER_WIDTH = 40
    PLAYER_HEIGHT = 20
    PLAYER_SCREEN_X = 100
    GRAVITY = 1.0
    JUMP_STRENGTH = 15.0

    # Game
    TRACK_Y = 320
    CAR_SPEED = 8.0 # pixels per frame
    MAX_STAGES = 3
    STAGE_LENGTH = 5000 # pixels
    STAGE_TIME_LIMIT = 60 # seconds

    # Obstacles
    OBSTACLE_MIN_WIDTH = 20
    OBSTACLE_MAX_WIDTH = 50
    OBSTACLE_MIN_HEIGHT = 20
    OBSTACLE_MAX_HEIGHT = 60
    OBSTACLE_SPAWN_BUFFER = 300 # min pixels between obstacles
    OBSTACLE_SPAWN_RANDOM = 400 # random extra pixels

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption(self.game_description)

        # Initialize state variables
        self.player_pos_y = 0
        self.player_vel_y = 0
        self.player_rect = pygame.Rect(0, 0, 0, 0)
        self.is_jumping = False
        self.space_was_held = False
        
        self.obstacles = []
        self.particles = []
        
        self.camera_x = 0
        self.stage = 1
        self.stage_progress = 0
        self.stage_timer = 0
        self.obstacle_speed_multiplier = 1.0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        # This will be called once in __init__
        self.reset()
        
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        # Player state
        self.player_pos_y = self.TRACK_Y - self.PLAYER_HEIGHT
        self.player_vel_y = 0
        self.is_jumping = False
        self.space_was_held = False
        self.player_rect = pygame.Rect(
            self.PLAYER_SCREEN_X, self.player_pos_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )
        
        # World state
        self.camera_x = 0
        self.stage = options.get("stage", 1) if options else 1
        self.stage_progress = 0
        self.stage_timer = self.STAGE_TIME_LIMIT
        self.obstacle_speed_multiplier = 1.0

        # Dynamic elements
        self.obstacles = []
        self.particles = []
        self._last_obstacle_spawn_pos = 0

        obs = self._get_observation()
        if self.render_mode == "human":
            self._render_to_human_screen(obs)

        return obs, self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # --- Handle Input ---
        movement = action[0] # unused
        space_held = action[1] == 1
        shift_held = action[2] == 1 # unused

        if space_held and not self.space_was_held and not self.is_jumping and not self.game_over:
            self.is_jumping = True
            self.player_vel_y = self.JUMP_STRENGTH
            self._spawn_particles(self.player_rect.midbottom, 15, self.COLOR_PARTICLE_JUMP, 'jump')
            # sfx: jump_sound()
        self.space_was_held = space_held

        if not self.game_over:
            # --- Update Game Logic ---
            self.steps += 1
            
            # Update world scroll
            self.camera_x += self.CAR_SPEED
            self.stage_progress += self.CAR_SPEED

            # Update player physics
            self._update_player()

            # Update obstacles
            self._update_obstacles()

            # Update particles
            self._update_particles()
            
            # Update timer and difficulty
            self.stage_timer -= 1 / self.FPS
            if self.steps > 0 and self.steps % 500 == 0:
                 self.obstacle_speed_multiplier += 0.05

            # --- Calculate Rewards ---
            # Survival reward
            reward += 0.1
            self.score += 0.1
            
            # Jump-over-obstacle reward
            for obs in self.obstacles:
                if not obs['cleared'] and self.player_rect.left > obs['rect'].right:
                    obs['cleared'] = True
                    reward += 1.0
                    self.score += 1.0

            # --- Check Termination Conditions ---
            # 1. Collision
            if self._check_collision():
                terminated = True
                self.game_over = True
                self.win_message = "CRASHED!"
                reward = -100.0
                self.score = -100.0
                self._spawn_particles(self.player_rect.center, 50, self.COLOR_PARTICLE_CRASH, 'crash')
                # sfx: crash_sound()
            
            # 2. Reached finish line
            if self.stage_progress >= self.STAGE_LENGTH:
                if self.stage == self.MAX_STAGES:
                    terminated = True
                    self.game_over = True
                    self.win_message = "YOU WIN!"
                    reward += 300.0
                    self.score += 300.0
                else:
                    reward += 100.0
                    self.score += 100.0
                    self._advance_to_next_stage()
                # sfx: win_sound()

            # 3. Timeout
            if self.stage_timer <= 0:
                terminated = True
                self.game_over = True
                self.win_message = "TIME OUT!"
                reward = -100.0
                self.score = -100.0
                # sfx: timeout_sound()

        # Update clock for auto-advance
        if self.auto_advance:
            self.clock.tick(self.FPS)

        obs = self._get_observation()
        if self.render_mode == "human":
            self._render_to_human_screen(obs)

        return (
            obs,
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self):
        if self.is_jumping:
            self.player_pos_y -= self.player_vel_y
            self.player_vel_y -= self.GRAVITY
        
        if self.player_pos_y >= self.TRACK_Y - self.PLAYER_HEIGHT:
            self.player_pos_y = self.TRACK_Y - self.PLAYER_HEIGHT
            self.player_vel_y = 0
            if self.is_jumping:
                self.is_jumping = False
                # sfx: land_sound()

        self.player_rect.y = int(self.player_pos_y)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['rect'].x -= int(self.CAR_SPEED * self.obstacle_speed_multiplier)

        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

        spawn_trigger_pos = self.camera_x + self.SCREEN_WIDTH
        if spawn_trigger_pos - self._last_obstacle_spawn_pos > self.OBSTACLE_SPAWN_BUFFER:
            self._spawn_obstacle()
            self._last_obstacle_spawn_pos = spawn_trigger_pos + self.np_random.integers(0, self.OBSTACLE_SPAWN_RANDOM)

    def _spawn_obstacle(self):
        w = self.np_random.integers(self.OBSTACLE_MIN_WIDTH, self.OBSTACLE_MAX_WIDTH + 1)
        h = self.np_random.integers(self.OBSTACLE_MIN_HEIGHT, self.OBSTACLE_MAX_HEIGHT + 1)
        x = int(self.camera_x + self.SCREEN_WIDTH + 50)
        y = self.TRACK_Y - h
        
        if x < self.STAGE_LENGTH:
            self.obstacles.append({'rect': pygame.Rect(x, y, w, h), 'cleared': False})

    def _check_collision(self):
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['rect']):
                return True
        return False

    def _advance_to_next_stage(self):
        self.stage += 1
        self.stage_progress = 0
        self.stage_timer = self.STAGE_TIME_LIMIT
        self.camera_x = 0
        self.obstacles = []
        self._last_obstacle_spawn_pos = 0

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _spawn_particles(self, pos, count, color, p_type):
        for _ in range(count):
            if p_type == 'jump':
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(2, 5)]
            elif p_type == 'crash':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 8)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 26),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        track_y_int = int(self.TRACK_Y)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, track_y_int), (self.SCREEN_WIDTH, track_y_int), 3)

        finish_x = self.STAGE_LENGTH - self.camera_x
        if finish_x < self.SCREEN_WIDTH + 50:
            for i in range(10):
                color = self.COLOR_FINISH_LINE if i % 2 == 0 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, (finish_x, self.TRACK_Y - 10*i, 20, 10))

        for obs in self.obstacles:
            draw_rect = obs['rect'].copy()
            draw_rect.x -= int(self.camera_x)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, draw_rect)
            pygame.draw.rect(self.screen, self.COLOR_TRACK, draw_rect, 1)

        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = max(1, int(p['life'] / 5))
            pygame.draw.rect(self.screen, p['color'], (pos[0], pos[1], size, size))
            
        glow_surface = pygame.Surface((self.PLAYER_WIDTH * 2, self.PLAYER_HEIGHT * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect())
        self.screen.blit(glow_surface, (self.player_rect.centerx - self.PLAYER_WIDTH, self.player_rect.centery - self.PLAYER_HEIGHT), special_flags=pygame.BLEND_RGBA_ADD)

        p = self.player_rect
        car_points = [
            (p.left, p.bottom), (p.left + 5, p.top + 5), (p.right - 10, p.top),
            (p.right, p.top + 10), (p.right, p.bottom)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, car_points)
        pygame.gfxdraw.aapolygon(self.screen, car_points, self.COLOR_PLAYER)

    def _render_ui(self):
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 10))
        
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        timer_color = self.COLOR_UI_FAIL if self.stage_timer < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_small.render(f"TIME: {max(0, int(self.stage_timer))}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        progress_ratio = min(1.0, self.stage_progress / self.STAGE_LENGTH)
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (10, 35, bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (10, 35, bar_width * progress_ratio, 10))

        if self.game_over:
            color = self.COLOR_UI_SUCCESS if "WIN" in self.win_message else self.COLOR_UI_FAIL
            msg_text = self.font_large.render(self.win_message, True, color)
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)

    def _render_to_human_screen(self, obs):
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        self.human_screen.blit(surf, (0, 0))
        pygame.display.flip()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "stage_progress": self.stage_progress,
            "time_left": self.stage_timer,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    terminated = False
    action = env.action_space.sample()
    action.fill(0)

    print("\n" + "="*40 + "\nGAME CONTROLS\n" + env.user_guide + "\n" + "="*40 + "\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        action.fill(0)
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_ESCAPE]:
            terminated = True

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(2000)

    env.close()