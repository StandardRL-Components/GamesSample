import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:07:53.488328
# Source Brief: brief_00892.md
# Brief Index: 892
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "A physics-based arcade game where you guide a frog across a busy highway. "
        "Time your jumps carefully to avoid traffic and reach the other side."
    )
    user_guide = (
        "Controls: Use arrow keys to aim your jump and nudge mid-air. Press space to jump."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.SIDEWALK_HEIGHT = 50
        self.LANE_COUNT = 4
        
        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_ROAD = (50, 50, 55)
        self.COLOR_SIDEWALK = (90, 95, 100)
        self.COLOR_LANE_MARKING = (200, 200, 180)
        self.COLOR_FROG = (50, 220, 50)
        self.COLOR_FROG_OUTLINE = (150, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)

        # --- Physics ---
        self.GRAVITY = 0.35
        self.FRICTION = 0.99
        self.JUMP_FORCE_Y = -9.0
        self.JUMP_FORCE_X = 4.0
        self.NUDGE_FORCE = 0.4
        
        # --- Game Parameters ---
        self.MAX_STEPS = 2000
        self.DIFFICULTY_INTERVAL = 500
        self.MAX_VEHICLE_SPEED = 3.0
        self.FROG_SIZE = 12

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.game_over = False
        self.frog_pos = np.array([0.0, 0.0])
        self.frog_vel = np.array([0.0, 0.0])
        self.frog_on_ground = True
        self.vehicles = []
        self.particles = []
        self.vehicle_base_speed = 1.0
        self.last_frog_y = 0.0

        # --- Lane & World Setup ---
        self.lane_height = (self.SCREEN_HEIGHT - 2 * self.SIDEWALK_HEIGHT) / self.LANE_COUNT
        self.lanes = []
        for i in range(self.LANE_COUNT):
            direction = 1 if i % 2 == 0 else -1
            y_pos = self.SIDEWALK_HEIGHT + self.lane_height * (i + 0.5)
            self.lanes.append({'y': y_pos, 'direction': direction})
        
        self.safe_zones = []
        for i in range(1, self.LANE_COUNT):
             y_pos = self.SIDEWALK_HEIGHT + self.lane_height * i
             self.safe_zones.append((y_pos - 4, y_pos + 4))

        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        
        self.frog_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.SIDEWALK_HEIGHT / 2], dtype=float)
        self.last_frog_y = self.frog_pos[1]
        self.frog_vel = np.array([0.0, 0.0], dtype=float)
        self.frog_on_ground = True
        
        self.vehicles = []
        self.particles = []
        self.vehicle_base_speed = 1.0
        self._spawn_initial_vehicles()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        self._handle_input(action)
        reward += self._update_frog()

        self._update_vehicles()
        self._update_particles()
        
        if self._check_collisions():
            self.game_over = True
            reward = -100.0
            # sfx: frog_squash.wav
            self._create_collision_particles(self.frog_pos)
        
        if not self.game_over and self.frog_pos[1] < self.SIDEWALK_HEIGHT:
            self.game_over = True
            reward = 100.0
            # sfx: win_jingle.wav
        
        self.steps += 1
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0 and self.vehicle_base_speed < self.MAX_VEHICLE_SPEED:
            self.vehicle_base_speed = min(self.MAX_VEHICLE_SPEED, self.vehicle_base_speed + 0.1)
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1

        if self.frog_on_ground and space_pressed:
            # sfx: jump.wav
            self.frog_on_ground = False
            if movement == 0: # None/Up
                self.frog_vel = np.array([0.0, self.JUMP_FORCE_Y])
            elif movement == 1: # Up
                self.frog_vel = np.array([0.0, self.JUMP_FORCE_Y * 1.1])
            elif movement == 2: # Down
                self.frog_vel = np.array([0.0, self.JUMP_FORCE_Y * 0.6])
            elif movement == 3: # Left
                self.frog_vel = np.array([-self.JUMP_FORCE_X, self.JUMP_FORCE_Y * 0.9])
            elif movement == 4: # Right
                self.frog_vel = np.array([self.JUMP_FORCE_X, self.JUMP_FORCE_Y * 0.9])
        elif not self.frog_on_ground:
            if movement == 1: self.frog_vel[1] -= self.NUDGE_FORCE
            elif movement == 2: self.frog_vel[1] += self.NUDGE_FORCE
            elif movement == 3: self.frog_vel[0] -= self.NUDGE_FORCE
            elif movement == 4: self.frog_vel[0] += self.NUDGE_FORCE
    
    def _update_frog(self):
        step_reward = 0
        was_on_ground = self.frog_on_ground

        if not self.frog_on_ground:
            self.frog_vel[1] += self.GRAVITY
        
        self.frog_vel *= self.FRICTION
        self.frog_pos += self.frog_vel

        y_progress = self.last_frog_y - self.frog_pos[1]
        if y_progress > 0:
            step_reward += y_progress * 0.1
        self.last_frog_y = self.frog_pos[1]

        landed = False
        if self.frog_vel[1] > 0:
            if self.frog_pos[1] >= self.SCREEN_HEIGHT - self.SIDEWALK_HEIGHT:
                self.frog_pos[1] = self.SCREEN_HEIGHT - self.SIDEWALK_HEIGHT
                landed = True
            elif not was_on_ground:
                for y_min, y_max in self.safe_zones:
                    if y_min <= self.frog_pos[1] <= y_max:
                        self.frog_pos[1] = (y_min + y_max) / 2
                        landed = True
                        step_reward += 1.0
                        # sfx: land_safe.wav
                        break
        
        if landed and not was_on_ground:
            self.frog_on_ground = True
            self.frog_vel = np.array([0.0, 0.0])
            # sfx: land_thud.wav

        self.frog_pos[0] = np.clip(self.frog_pos[0], self.FROG_SIZE, self.SCREEN_WIDTH - self.FROG_SIZE)
        if self.frog_pos[1] > self.SCREEN_HEIGHT - self.FROG_SIZE:
            self.frog_pos[1] = self.SCREEN_HEIGHT - self.FROG_SIZE
            self.frog_vel[1] = 0
        
        return step_reward
    
    def _update_vehicles(self):
        for v in self.vehicles[:]:
            v['rect'].x += v['speed']
            if (v['speed'] > 0 and v['rect'].left > self.SCREEN_WIDTH) or \
               (v['speed'] < 0 and v['rect'].right < 0):
                self.vehicles.remove(v)

        for i, lane in enumerate(self.lanes):
            if self.np_random.random() < 0.025:
                self._spawn_vehicle(i)

    def _spawn_initial_vehicles(self):
        for i in range(self.LANE_COUNT):
            for _ in range(self.np_random.integers(1, 4)):
                self._spawn_vehicle(i, x_pos=self.np_random.integers(0, self.SCREEN_WIDTH))

    def _spawn_vehicle(self, lane_index, x_pos=None):
        lane = self.lanes[lane_index]
        is_truck = self.np_random.random() < 0.2
        width = 60 if is_truck else 40
        height = 30
        
        if x_pos is None:
            x = -width if lane['direction'] > 0 else self.SCREEN_WIDTH
        else:
            x = x_pos
        
        new_rect = pygame.Rect(int(x), int(lane['y'] - height / 2), width, height)
        
        # Prevent immediate overlap on spawn
        for v in self.vehicles:
            if v['rect'].colliderect(new_rect.inflate(80, 0)):
                return

        speed_multiplier = self.np_random.uniform(0.8, 1.2)
        speed = lane['direction'] * self.vehicle_base_speed * speed_multiplier
        color = (self.np_random.integers(100, 201), self.np_random.integers(100, 201), self.np_random.integers(100, 201))
        if is_truck:
            color = (self.np_random.integers(50, 101), self.np_random.integers(50, 101), self.np_random.integers(150, 221))

        self.vehicles.append({'rect': new_rect, 'speed': speed, 'color': color})

    def _check_collisions(self):
        frog_hitbox = pygame.Rect(0, 0, self.FROG_SIZE, self.FROG_SIZE)
        frog_hitbox.center = (int(self.frog_pos[0]), int(self.frog_pos[1]))
        for v in self.vehicles:
            if frog_hitbox.colliderect(v['rect']):
                return True
        return False

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_collision_particles(self, pos):
        for _ in range(40):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': self.np_random.integers(20, 41),
                'radius': self.np_random.uniform(2, 6),
                'color': random.choice([self.COLOR_FROG, (255, 50, 50), (255, 255, 100)])
            })

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"steps": self.steps}

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_SIDEWALK, (0, 0, self.SCREEN_WIDTH, self.SIDEWALK_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_SIDEWALK, (0, self.SCREEN_HEIGHT - self.SIDEWALK_HEIGHT, self.SCREEN_WIDTH, self.SIDEWALK_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_ROAD, (0, self.SIDEWALK_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 2 * self.SIDEWALK_HEIGHT))
        
        for i in range(1, self.LANE_COUNT):
            y = self.SIDEWALK_HEIGHT + i * self.lane_height
            for x in range(0, self.SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, self.COLOR_LANE_MARKING, (x, y), (x + 20, y), 2)
        
        self._render_vehicles()
        if not self.game_over:
            self._render_frog()
        self._render_particles()

    def _render_ui(self):
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))

    def _render_vehicles(self):
        for v in self.vehicles:
            pygame.gfxdraw.box(self.screen, v['rect'], v['color'])
            darker_color = tuple(max(0, c - 40) for c in v['color'])
            pygame.gfxdraw.rectangle(self.screen, v['rect'], darker_color)

            light_color = (255, 255, 200) if v['speed'] > 0 else (255, 50, 50)
            light_size = 6
            if v['speed'] > 0:
                pygame.draw.rect(self.screen, light_color, (v['rect'].right - 3, v['rect'].top + 5, 4, light_size))
                pygame.draw.rect(self.screen, light_color, (v['rect'].right - 3, v['rect'].bottom - 5 - light_size, 4, light_size))
            else:
                pygame.draw.rect(self.screen, light_color, (v['rect'].left - 1, v['rect'].top + 5, 4, light_size))
                pygame.draw.rect(self.screen, light_color, (v['rect'].left - 1, v['rect'].bottom - 5 - light_size, 4, light_size))

    def _render_frog(self):
        x, y = int(self.frog_pos[0]), int(self.frog_pos[1])
        
        stretch_v = np.clip(self.frog_vel[1] / self.JUMP_FORCE_Y, -0.7, 1.0)
        squash_h = np.clip(abs(self.frog_vel[0]) / (self.JUMP_FORCE_X + 1e-6), 0, 0.5)
        
        w = int(self.FROG_SIZE * (1 + stretch_v * 0.5 + squash_h))
        h = int(self.FROG_SIZE * (1 - stretch_v * 0.8))

        glow_radius = int(self.FROG_SIZE * 1.6)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = (*self.COLOR_FROG_OUTLINE, 80)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        if w > 0 and h > 0:
            pygame.gfxdraw.filled_ellipse(self.screen, x, y, w, h, self.COLOR_FROG)
            pygame.gfxdraw.ellipse(self.screen, x, y, w, h, tuple(max(0, c - 40) for c in self.COLOR_FROG))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] * (p['life'] / 40.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, radius), color)
    
    def close(self):
        pygame.quit()
        
    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the autograder.
    # Set the SDL_VIDEODRIVER to a real driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Physics Frogger")
    clock = pygame.time.Clock()
    
    total_reward = 0.0
    
    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # none
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and done:
                    obs, info = env.reset()
                    total_reward = 0.0
                    done = False
                if event.key == pygame.K_q:
                    running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
        
        action = [movement, space_pressed, shift_pressed]
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        
        if done:
            font_large = pygame.font.Font(None, 72)
            font_small = pygame.font.Font(None, 36)
            
            end_text = "YOU WON!" if total_reward > 0 else "GAME OVER"
            text_surf = font_large.render(end_text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 - 20))
            render_screen.blit(text_surf, text_rect)
            
            reward_text = font_small.render(f"Final Reward: {total_reward:.2f}", True, (255, 255, 255))
            reward_rect = reward_text.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 + 30))
            render_screen.blit(reward_text, reward_rect)

            reset_text = font_small.render("Press 'R' to play again", True, (200, 200, 200))
            reset_rect = reset_text.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 + 70))
            render_screen.blit(reset_text, reset_rect)

        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])
        
    env.close()