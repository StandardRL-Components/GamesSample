
# Generated: 2025-08-27T20:50:53.634454
# Source Brief: brief_02594.md
# Brief Index: 2594

        
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
        "Controls: Arrow keys to move. Press Space to dash when stamina is full."
    )

    game_description = (
        "Survive the night in a haunted graveyard, collecting lost souls while evading a relentless ghost until 6:00 AM."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 1280, 800
        self.FPS = 30

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_SOUL = (100, 255, 150)
        self.COLOR_MONSTER = (220, 220, 255)
        self.COLOR_MONSTER_HUNT = (255, 50, 50)
        self.COLOR_OBSTACLE = (50, 55, 70)
        self.COLOR_MAUSOLEUM = (40, 45, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_STAMINA_BAR = (0, 180, 255)
        self.COLOR_STAMINA_BG = (50, 50, 80)

        # Game Mechanics
        self.PLAYER_SPEED = 3.0
        self.PLAYER_DASH_SPEED = 9.0
        self.PLAYER_DASH_DURATION = 0.2 * self.FPS  # 0.2 seconds
        self.PLAYER_STAMINA_MAX = 1.0
        self.PLAYER_STAMINA_REGEN = 0.01

        self.MONSTER_PATROL_SPEED = 1.8
        self.MONSTER_HUNT_SPEED = 3.5
        self.MONSTER_SIGHT_RADIUS = 200
        self.MONSTER_CATCH_RADIUS = 20
        self.MONSTER_SPEED_INCREASE_INTERVAL = 60 # seconds
        self.MONSTER_SPEED_INCREMENT = 0.2

        self.MAX_SOULS = 10
        self.SOUL_COLLECT_RADIUS = 20
        self.MAX_OBSTACLES = 25

        self.GAME_DURATION_SECONDS = 360  # 6 in-game hours (1 min/sec)

        # Rewards
        self.REWARD_SURVIVE_SECOND = 0.1
        self.REWARD_COLLECT_SOUL = 10.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Impact", 60)
        
        # --- State variables ---
        self.player_pos = None
        self.player_vel = None
        self.player_stamina = None
        self.is_dashing = None
        self.dash_timer = None
        self.prev_space_held = None

        self.monster_pos = None
        self.monster_vel = None
        self.monster_target = None
        self.monster_state = None # "PATROL" or "HUNT"

        self.souls = None
        self.obstacles = None
        self.mausoleum_rect = None
        self.camera_pos = None
        
        self.steps = 0
        self.score = 0
        self.game_time_seconds = 0
        self.terminated = False

        self.reset()
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_time_seconds = 0
        self.terminated = False

        # Player
        self.player_pos = pygame.math.Vector2(self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_stamina = self.PLAYER_STAMINA_MAX
        self.is_dashing = False
        self.dash_timer = 0
        self.prev_space_held = False
        
        # Monster
        self.monster_pos = self._get_random_spawn_pos(avoid_pos=self.player_pos, min_dist=300)
        self.monster_vel = pygame.math.Vector2(0, 0)
        self.monster_state = "PATROL"
        self.monster_target = self._get_random_spawn_pos()
        
        # World
        self._generate_layout()
        self.souls = [self._get_random_spawn_pos() for _ in range(self.MAX_SOULS)]
        
        self.camera_pos = pygame.math.Vector2(self.player_pos.x, self.player_pos.y)

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1
        self.game_time_seconds += 1 / self.FPS
        reward = self.REWARD_SURVIVE_SECOND / self.FPS

        self._handle_input(action)
        self._update_player()
        self._update_monster()
        
        reward += self._handle_interactions()
        
        self.terminated = self._check_termination()

        if self.terminated:
            if self.game_time_seconds >= self.GAME_DURATION_SECONDS:
                reward += self.REWARD_WIN
            else:
                reward += self.REWARD_LOSE

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Movement
        move_direction = pygame.math.Vector2(0, 0)
        if not self.is_dashing:
            if movement == 1: move_direction.y = -1
            elif movement == 2: move_direction.y = 1
            elif movement == 3: move_direction.x = -1
            elif movement == 4: move_direction.x = 1

            if move_direction.length() > 0:
                self.player_vel = move_direction.normalize() * self.PLAYER_SPEED
            else:
                self.player_vel = pygame.math.Vector2(0, 0)
        
        # Dash
        if space_held and not self.prev_space_held and self.player_stamina >= self.PLAYER_STAMINA_MAX and not self.is_dashing:
            # sfx: player_dash.wav
            self.is_dashing = True
            self.dash_timer = self.PLAYER_DASH_DURATION
            self.player_stamina = 0
            if self.player_vel.length() > 0:
                self.player_vel = self.player_vel.normalize() * self.PLAYER_DASH_SPEED
            else: # Dash forward if standing still
                self.player_vel = pygame.math.Vector2(0, -1) * self.PLAYER_DASH_SPEED
        
        self.prev_space_held = space_held

    def _update_player(self):
        # Dash logic
        if self.is_dashing:
            self.dash_timer -= 1
            if self.dash_timer <= 0:
                self.is_dashing = False
        
        # Stamina regen
        if not self.is_dashing and self.player_stamina < self.PLAYER_STAMINA_MAX:
            self.player_stamina = min(self.PLAYER_STAMINA_MAX, self.player_stamina + self.PLAYER_STAMINA_REGEN)

        # Update position and handle collisions
        new_pos = self.player_pos + self.player_vel
        self._collide_with_obstacles(self.player_pos, self.player_vel, 10)
        self.player_pos += self.player_vel
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WORLD_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.WORLD_HEIGHT)

    def _update_monster(self):
        player_is_safe = self.mausoleum_rect.collidepoint(self.player_pos.x, self.player_pos.y)
        dist_to_player = self.player_pos.distance_to(self.monster_pos)

        # AI State
        if not player_is_safe and dist_to_player < self.MONSTER_SIGHT_RADIUS:
            if self.monster_state == "PATROL": # sfx: monster_aggro.wav
                pass
            self.monster_state = "HUNT"
            self.monster_target = self.player_pos
        else:
            self.monster_state = "PATROL"
            if self.monster_pos.distance_to(self.monster_target) < 50:
                self.monster_target = self._get_random_spawn_pos()

        # Speed
        time_based_speed_increase = (self.game_time_seconds // self.MONSTER_SPEED_INCREASE_INTERVAL) * self.MONSTER_SPEED_INCREMENT
        if self.monster_state == "HUNT":
            speed = self.MONSTER_HUNT_SPEED + time_based_speed_increase
        else:
            speed = self.MONSTER_PATROL_SPEED + time_based_speed_increase

        # Movement
        direction = (self.monster_target - self.monster_pos)
        if direction.length() > 0:
            self.monster_vel = direction.normalize() * speed
        else:
            self.monster_vel = pygame.math.Vector2(0, 0)
        
        # Update position and handle collisions
        self._collide_with_obstacles(self.monster_pos, self.monster_vel, 15)
        self.monster_pos += self.monster_vel
        self.monster_pos.x = np.clip(self.monster_pos.x, 0, self.WORLD_WIDTH)
        self.monster_pos.y = np.clip(self.monster_pos.y, 0, self.WORLD_HEIGHT)
    
    def _handle_interactions(self):
        reward = 0
        
        # Soul Collection
        collected_souls = []
        for i, soul_pos in enumerate(self.souls):
            if self.player_pos.distance_to(soul_pos) < self.SOUL_COLLECT_RADIUS:
                collected_souls.append(i)
                self.score += 1
                reward += self.REWARD_COLLECT_SOUL
                # sfx: soul_collect.wav
        
        if collected_souls:
            self.souls = [s for i, s in enumerate(self.souls) if i not in collected_souls]
            for _ in range(len(collected_souls)):
                self.souls.append(self._get_random_spawn_pos())

        # Monster Catch
        player_is_safe = self.mausoleum_rect.collidepoint(self.player_pos.x, self.player_pos.y)
        if not player_is_safe and self.player_pos.distance_to(self.monster_pos) < self.MONSTER_CATCH_RADIUS:
            self.terminated = True
            # sfx: player_death.wav

        return reward

    def _check_termination(self):
        return self.terminated or self.game_time_seconds >= self.GAME_DURATION_SECONDS

    def _get_observation(self):
        self._update_camera()
        self.screen.fill(self.COLOR_BG)

        # Render world elements (offset by camera)
        self._render_world()
        
        # Render fog of war
        self._render_fog()

        # Render UI (static on screen)
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_time": self.game_time_seconds,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "monster_pos": (self.monster_pos.x, self.monster_pos.y),
        }

    def _update_camera(self):
        self.camera_pos.x = self.player_pos.x - self.WIDTH / 2
        self.camera_pos.y = self.player_pos.y - self.HEIGHT / 2
        self.camera_pos.x = np.clip(self.camera_pos.x, 0, self.WORLD_WIDTH - self.WIDTH)
        self.camera_pos.y = np.clip(self.camera_pos.y, 0, self.WORLD_HEIGHT - self.HEIGHT)

    def _render_world(self):
        # Mausoleum
        mausoleum_screen_rect = self.mausoleum_rect.move(-self.camera_pos.x, -self.camera_pos.y)
        pygame.draw.rect(self.screen, self.COLOR_MAUSOLEUM, mausoleum_screen_rect)

        # Obstacles
        for obs_rect in self.obstacles:
            obs_screen_rect = obs_rect.move(-self.camera_pos.x, -self.camera_pos.y)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_screen_rect, border_radius=3)
        
        # Souls
        for soul_pos in self.souls:
            sx, sy = int(soul_pos.x - self.camera_pos.x), int(soul_pos.y - self.camera_pos.y)
            self._draw_glow_circle(self.screen, sx, sy, 8, self.COLOR_SOUL, 20)

        # Monster
        mx, my = int(self.monster_pos.x - self.camera_pos.x), int(self.monster_pos.y - self.camera_pos.y)
        monster_color = self.COLOR_MONSTER_HUNT if self.monster_state == "HUNT" else self.COLOR_MONSTER
        self._draw_glow_circle(self.screen, mx, my, 15, monster_color, 30, 100)
        
        # Player
        px, py = int(self.player_pos.x - self.camera_pos.x), int(self.player_pos.y - self.camera_pos.y)
        self._draw_glow_circle(self.screen, px, py, 10, self.COLOR_PLAYER, 25)

    def _render_fog(self):
        fog_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        fog_surface.fill((0, 0, 0, 210)) # Dark fog
        
        # Player light source
        px, py = int(self.player_pos.x - self.camera_pos.x), int(self.player_pos.y - self.camera_pos.y)
        light_radius = 220
        pygame.draw.circle(fog_surface, (0, 0, 0, 0), (px, py), light_radius)
        
        # Monster glow through fog
        if self.monster_state == "HUNT":
            mx, my = int(self.monster_pos.x - self.camera_pos.x), int(self.monster_pos.y - self.camera_pos.y)
            monster_glow_radius = 40
            pygame.draw.circle(fog_surface, (150, 0, 0, 150), (mx, my), monster_glow_radius, 0)
            
        self.screen.blit(fog_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    def _render_ui(self):
        # Soul Counter
        soul_text = self.font_ui.render(f"Souls: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(soul_text, (10, 10))

        # Clock
        minutes = int(self.game_time_seconds / 60) % 60
        hours = int(minutes / 60)
        display_time = f"{hours:02d}:{minutes:02d} AM"
        time_text = self.font_ui.render(display_time, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Stamina Bar
        bar_width = 100
        bar_height = 10
        pygame.draw.rect(self.screen, self.COLOR_STAMINA_BG, (self.WIDTH / 2 - bar_width / 2, self.HEIGHT - 25, bar_width, bar_height))
        fill_width = self.player_stamina * bar_width
        pygame.draw.rect(self.screen, self.COLOR_STAMINA_BAR, (self.WIDTH / 2 - bar_width / 2, self.HEIGHT - 25, fill_width, bar_height))

        # Game Over / Win Text
        if self.terminated:
            if self.game_time_seconds >= self.GAME_DURATION_SECONDS:
                msg = "YOU SURVIVED"
                color = self.COLOR_PLAYER
            else:
                msg = "YOU WERE CAUGHT"
                color = self.COLOR_MONSTER_HUNT
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _generate_layout(self):
        self.mausoleum_rect = pygame.Rect(self.WORLD_WIDTH/2 - 75, self.WORLD_HEIGHT/2 - 100, 150, 200)
        self.obstacles = []
        for _ in range(self.MAX_OBSTACLES):
            w = self.np_random.integers(20, 60)
            h = self.np_random.integers(40, 80)
            pos = self._get_random_spawn_pos()
            rect = pygame.Rect(pos.x, pos.y, w, h)
            if not rect.colliderect(self.mausoleum_rect):
                self.obstacles.append(rect)
    
    def _get_random_spawn_pos(self, avoid_pos=None, min_dist=0):
        while True:
            pos = pygame.math.Vector2(
                self.np_random.uniform(20, self.WORLD_WIDTH - 20),
                self.np_random.uniform(20, self.WORLD_HEIGHT - 20)
            )
            # Check if inside an obstacle
            in_obstacle = any(obs.collidepoint(pos.x, pos.y) for obs in self.obstacles) if self.obstacles else False
            in_mausoleum = self.mausoleum_rect.collidepoint(pos.x, pos.y) if self.mausoleum_rect else False
            too_close = (avoid_pos is not None and pos.distance_to(avoid_pos) < min_dist)

            if not in_obstacle and not in_mausoleum and not too_close:
                return pos

    def _collide_with_obstacles(self, pos, vel, radius):
        # Simple rect-based collision
        temp_rect = pygame.Rect(pos.x - radius, pos.y - radius, radius*2, radius*2)
        
        # Horizontal collision
        temp_rect.x += vel.x
        for obs in self.obstacles + [self.mausoleum_rect]:
            if temp_rect.colliderect(obs):
                if vel.x > 0: temp_rect.right = obs.left
                if vel.x < 0: temp_rect.left = obs.right
                vel.x = 0
        
        # Vertical collision
        temp_rect.y += vel.y
        for obs in self.obstacles + [self.mausoleum_rect]:
            if temp_rect.colliderect(obs):
                if vel.y > 0: temp_rect.bottom = obs.top
                if vel.y < 0: temp_rect.top = obs.bottom
                vel.y = 0

    def _draw_glow_circle(self, surface, x, y, radius, color, glow_radius, glow_alpha=50):
        # Draw soft glow
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Draw main circle
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, color)
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
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
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Haunted Graveyard")
    
    running = True
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Time Survived: {info['game_time']:.2f}s")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()