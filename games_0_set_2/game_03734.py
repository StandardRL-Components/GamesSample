import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press Space to jump. Avoid obstacles and don't fall off the line. Reach the red finish line before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race against time on a procedurally generated track, using limited jumps to overcome obstacles and reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Fonts ---
        self.UI_FONT = None  # Use Pygame's default font
        self.TITLE_FONT = None
        self.COLOR_BG_TOP = (20, 25, 40)
        self.COLOR_BG_BOTTOM = (40, 50, 80)
        self.COLOR_TRACK = (20, 20, 20)
        self.COLOR_OBSTACLE = (100, 110, 120)
        self.COLOR_START = (0, 200, 100)
        self.COLOR_FINISH = (220, 50, 50)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 255, 40)
        self.COLOR_UI = (230, 230, 230)
        self.COLOR_WIN_TEXT = (100, 255, 150)
        self.COLOR_LOSE_TEXT = (255, 100, 100)

        # --- Game Constants ---
        self.MAX_TIME_SECONDS = 60
        self.FPS = 30
        self.INITIAL_JUMPS = 3
        self.MAX_STAGES = 3

        # --- Physics Constants ---
        self.GRAVITY = 0.4
        self.JUMP_VELOCITY = -9
        self.PLAYER_SPEED = 5.0

        # --- Reward Constants ---
        self.REWARD_PER_STEP = 0.01
        self.REWARD_CLEAR_OBSTACLE = 5.0
        self.REWARD_WASTE_JUMP = -1.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0
        
        # --- State Variables ---
        self.np_random = None
        self.stage = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.time_left_frames = 0
        self.jumps_left = 0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_in_air = True
        self.camera_x = 0
        self.track_points = []
        self.obstacles = []
        self.finish_line_x = 0
        self.prev_space_held = False
        self.jump_just_initiated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.stage = options.get("stage", 1) if options else 1
        self.stage = max(1, min(self.stage, self.MAX_STAGES))

        self._setup_stage(self.stage)
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _setup_stage(self, stage_num):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.time_left_frames = self.MAX_TIME_SECONDS * self.FPS
        self.jumps_left = self.INITIAL_JUMPS
        
        self.player_pos = pygame.Vector2(100, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_in_air = True
        self.prev_space_held = False
        self.jump_just_initiated = False

        self._generate_track_and_obstacles(stage_num)
        
        initial_track_y = self._get_track_y_at(self.player_pos.x)
        if initial_track_y is not None:
            self.player_pos.y = initial_track_y
            self.player_in_air = False
        
        self.camera_x = self.player_pos.x - self.WIDTH / 4

    def _generate_track_and_obstacles(self, stage_num):
        self.track_points = []
        num_segments = 150 + stage_num * 25
        segment_length = 60

        y = self.HEIGHT * 0.7
        for i in range(num_segments):
            self.track_points.append(pygame.Vector2(i * segment_length, y))
            y = self.HEIGHT * 0.7 + math.sin(i / 20) * 60 + math.sin(i / 7) * 30 + self.np_random.uniform(-5, 5)
            y = np.clip(y, self.HEIGHT * 0.3, self.HEIGHT * 0.9)

        self.finish_line_x = self.track_points[-10].x

        self.obstacles = []
        num_obstacles = 5 * stage_num
        obstacle_width = 20
        obstacle_height = 40
        
        possible_indices = list(range(10, len(self.track_points) - 15))
        if possible_indices: # Ensure list is not empty
            self.np_random.shuffle(possible_indices)
            
            for i in range(min(num_obstacles, len(possible_indices))):
                idx = possible_indices[i]
                p1 = self.track_points[idx]
                self.obstacles.append({
                    "rect": pygame.Rect(p1.x - obstacle_width / 2, p1.y - obstacle_height, obstacle_width, obstacle_height),
                    "cleared": False
                })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        space_held = action[1] == 1
        
        self.jump_just_initiated = False
        if space_held and not self.prev_space_held and self.jumps_left > 0 and not self.player_in_air:
            self.player_vel.y = self.JUMP_VELOCITY
            self.player_in_air = True
            self.jumps_left -= 1
            self.jump_just_initiated = True
        self.prev_space_held = space_held

        reward = self._update_game_state()
        self.steps += 1
        self.score += reward
        
        terminated = self.game_over
        
        if self.player_pos.x >= self.finish_line_x and not terminated:
            self.game_over = True
            self.victory = True
            terminated = True
            final_reward = self.REWARD_WIN
            reward += final_reward
            self.score += final_reward

        obs = self._get_observation()
        info = self._get_info()
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def _update_game_state(self):
        self.time_left_frames -= 1
        
        if self.player_in_air:
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel
        else:
            self.player_pos.x += self.PLAYER_SPEED
            track_y = self._get_track_y_at(self.player_pos.x)
            if track_y is not None:
                self.player_pos.y = track_y
            else:
                self.game_over = True

        self.camera_x = self.player_pos.x - self.WIDTH / 4

        reward = self.REWARD_PER_STEP
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 10, 10, 10)

        if self.player_in_air:
            track_y = self._get_track_y_at(self.player_pos.x)
            if track_y is not None and self.player_pos.y >= track_y and self.player_vel.y > 0:
                self.player_pos.y = track_y
                self.player_in_air = False
                self.player_vel.y = 0
        
        track_y_under_player = self._get_track_y_at(self.player_pos.x)
        if track_y_under_player is None or self.player_pos.y > self.HEIGHT:
            self.game_over = True

        for obs in self.obstacles:
            if player_rect.colliderect(obs["rect"]):
                if not self.player_in_air:
                    self.game_over = True
            
            if not obs["cleared"] and self.player_pos.x > obs["rect"].centerx:
                obs["cleared"] = True
                reward += self.REWARD_CLEAR_OBSTACLE
        
        if self.jump_just_initiated:
            is_necessary = False
            for obs in self.obstacles:
                if 0 < obs["rect"].centerx - self.player_pos.x < self.PLAYER_SPEED * 20:
                    is_necessary = True
                    break
            if not is_necessary:
                reward += self.REWARD_WASTE_JUMP

        if self.time_left_frames <= 0:
            self.game_over = True
        
        if self.game_over and not self.victory:
            reward += self.REWARD_LOSE

        return reward

    def _get_track_y_at(self, x):
        if not self.track_points or x < self.track_points[0].x or x > self.track_points[-1].x:
            return None
            
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            if p1.x <= x < p2.x:
                t = (x - p1.x) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
                return p1.y + t * (p2.y - p1.y)
        return self.track_points[-1].y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_TOP)
        bottom_rect = pygame.Rect(0, self.HEIGHT / 2, self.WIDTH, self.HEIGHT / 2)
        pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, bottom_rect)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        start_screen_x = int(self.track_points[0].x - self.camera_x)
        pygame.draw.line(self.screen, self.COLOR_START, (start_screen_x, 0), (start_screen_x, self.HEIGHT), 5)
        
        finish_screen_x = int(self.finish_line_x - self.camera_x)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.HEIGHT), 5)

        screen_points = []
        for p in self.track_points:
            screen_x = p.x - self.camera_x
            if -100 < screen_x < self.WIDTH + 100:
                screen_points.append((int(screen_x), int(p.y)))
        
        if len(screen_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, screen_points, 3)

        for obs in self.obstacles:
            screen_rect = obs["rect"].copy()
            screen_rect.x -= self.camera_x
            if screen_rect.right > 0 and screen_rect.left < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                pygame.draw.rect(self.screen, self.COLOR_TRACK, screen_rect, 1)

        if self.prev_space_held and not self.player_in_air and self.jumps_left > 0:
            # FIX: pygame.Vector2 does not have a .copy() method.
            # Create a new vector to avoid modifying the original.
            sim_pos = pygame.Vector2(self.player_pos)
            sim_vel = pygame.Vector2(self.PLAYER_SPEED, self.JUMP_VELOCITY)
            for i in range(20):
                sim_vel.y += self.GRAVITY
                sim_pos += sim_vel
                if i % 2 == 0:
                    screen_pos = (int(sim_pos.x - self.camera_x), int(sim_pos.y))
                    if 0 <= screen_pos[0] < self.WIDTH and 0 <= screen_pos[1] < self.HEIGHT:
                        pygame.gfxdraw.pixel(self.screen, screen_pos[0], screen_pos[1], self.COLOR_PLAYER)

        player_screen_pos = (int(self.player_pos.x - self.camera_x), int(self.player_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], 10, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, player_screen_pos[0], player_screen_pos[1], 10, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], 7, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_screen_pos[0], player_screen_pos[1], 7, self.COLOR_PLAYER)

    def _render_ui(self):
        time_text = f"TIME: {max(0, self.time_left_frames / self.FPS):.2f}"
        self._draw_text(time_text, self.UI_FONT, 24, self.COLOR_UI, (10, 10), "topleft")

        jumps_text = f"JUMPS: {self.jumps_left}"
        self._draw_text(jumps_text, self.UI_FONT, 24, self.COLOR_UI, (self.WIDTH - 10, 10), "topright")

        stage_text = f"STAGE {self.stage}"
        self._draw_text(stage_text, self.UI_FONT, 20, self.COLOR_UI, (self.WIDTH / 2, self.HEIGHT - 15), "midbottom")

        if self.game_over:
            msg, color = ("STAGE COMPLETE!", self.COLOR_WIN_TEXT) if self.victory else ("GAME OVER", self.COLOR_LOSE_TEXT)
            self._draw_text(msg, self.TITLE_FONT, 60, color, (self.WIDTH / 2, self.HEIGHT / 2), "center", shadow=True)

    def _draw_text(self, text, font_name, size, color, pos, anchor="topleft", shadow=False):
        try:
            font = pygame.font.Font(font_name, size)
            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect(**{anchor: pos})
            
            if shadow:
                shadow_surface = font.render(text, True, (0,0,0))
                shadow_rect = shadow_surface.get_rect(**{anchor: (pos[0]+2, pos[1]+2)})
                self.screen.blit(shadow_surface, shadow_rect)

            self.screen.blit(text_surface, text_rect)
        except pygame.error as e:
            # Silently fail on font errors in headless mode if default font isn't found
            pass

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": round(max(0, self.time_left_frames / self.FPS), 2),
            "jumps_left": self.jumps_left,
            "victory": self.victory,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("GameEnv")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    while not terminated:
        action = [0, 0, 0] 
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated:
            print(f"Episode finished. Score: {info['score']:.2f}, Victory: {info['victory']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()