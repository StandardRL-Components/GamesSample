import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:44:50.404153
# Source Brief: brief_03205.md
# Brief Index: 3205
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gravity Well: An arcade-style Gymnasium environment where the player
    activates gravity wells to capture and transform asteroids for points
    against a time limit.
    
    Action Space: MultiDiscrete([5, 2, 2])
    - action[0] (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right. Activates the corresponding gravity well.
    - action[1] (Space): Unused.
    - action[2] (Shift): Unused.
    
    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Activate gravity wells to pull asteroids into the central capture point. "
        "Hold asteroids in a well's influence to transform them for more points before time runs out."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to activate the corresponding gravity well. "
        "Lure asteroids into the central point to score."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors (Sci-fi, high contrast)
    COLOR_BG = (15, 20, 30)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CAPTURE_POINT = (255, 255, 255)
    COLOR_CAPTURE_GLOW = (255, 255, 255, 30)
    COLOR_WELL_INACTIVE = (50, 60, 80)
    COLOR_WELL_ACTIVE = (255, 80, 80)
    COLOR_WELL_PULSE = (255, 150, 150)
    COLOR_ASTEROID = (80, 160, 255)
    COLOR_ASTEROID_TRANSFORMED = (80, 255, 160)

    # Game Mechanics
    CAPTURE_POINT_POS = pygame.math.Vector2(WIDTH // 2, HEIGHT // 2)
    CAPTURE_POINT_RADIUS = 25
    GRAVITY_WELL_RADIUS = 100
    GRAVITY_STRENGTH = 150.0
    TRANSFORM_TIME_STEPS = 5 * FPS # 5 seconds

    ASTEROID_RADIUS_INITIAL = 10
    ASTEROID_SPEED_INITIAL = 1.0
    ASTEROID_SPAWN_FREQ_INITIAL_HZ = 1.0 / 3.0 # 1 every 3 seconds
    ASTEROID_SPAWN_FREQ_INCREASE_RATE = 0.05 # per 10 seconds
    ASTEROID_SPEED_INCREASE_RATE = 0.1 # per 10 seconds
    ASTEROID_SPAWN_FREQ_MAX_HZ = 2.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.Font(pygame.font.match_font('consolas,dejavusansmono,monospace'), 24)
            self.font_small = pygame.font.Font(pygame.font.match_font('consolas,dejavusansmono,monospace'), 18)
        except:
            self.font = pygame.font.SysFont('monospace', 24)
            self.font_small = pygame.font.SysFont('monospace', 18)

        # Game entities and state are initialized in reset()
        self.asteroids = []
        self.particles = []
        self.wells_active = [False] * 4
        self.well_activation_visual_timer = 0
        
        margin = 80
        self.well_positions = [
            pygame.math.Vector2(self.WIDTH / 2, margin),          # Top
            pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - margin), # Bottom
            pygame.math.Vector2(margin, self.HEIGHT / 2),          # Left
            pygame.math.Vector2(self.WIDTH - margin, self.HEIGHT / 2) # Right
        ]
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.asteroids = []
        self.particles = []
        self.wells_active = [False] * 4
        self.well_activation_visual_timer = 0
        
        self.current_asteroid_speed = self.ASTEROID_SPEED_INITIAL
        self.current_spawn_freq_hz = self.ASTEROID_SPAWN_FREQ_INITIAL_HZ
        self.asteroid_spawn_timer = self.FPS / self.current_spawn_freq_hz

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action
        self.steps += 1
        
        self._handle_input(movement)
        
        reward = self._update_game_state()
        
        self.game_over = self.steps >= self.MAX_STEPS
        terminated = self.game_over
        truncated = False # No truncation condition other than termination
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement_action):
        self.wells_active = [False] * 4
        if 1 <= movement_action <= 4:
            # action 1=up, 2=down, 3=left, 4=right maps to well_positions index 0,1,2,3
            self.wells_active[movement_action - 1] = True

    def _update_game_state(self):
        self.well_activation_visual_timer += 1
        
        self._update_difficulty()
        self._spawn_asteroids()
        
        # Update particles and remove dead ones
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Update asteroids
        reward = 0
        for asteroid in self.asteroids[:]:
            # Continuous reward for being near a well
            for well_pos in self.well_positions:
                if asteroid['pos'].distance_to(well_pos) < self.GRAVITY_WELL_RADIUS:
                    reward += 0.01
            
            self._update_one_asteroid(asteroid)
            
            # Check for capture
            if asteroid['pos'].distance_to(self.CAPTURE_POINT_POS) < self.CAPTURE_POINT_RADIUS + asteroid['radius']:
                if asteroid['transformed']:
                    self.score += 20
                    reward += 20
                else:
                    self.score += 10
                    reward += 10
                
                self._create_capture_particles(asteroid['pos'], asteroid['transformed'])
                self.asteroids.remove(asteroid)
        
        return reward

    def _update_one_asteroid(self, asteroid):
        # Apply gravity from active wells
        total_force = pygame.math.Vector2(0, 0)
        is_in_well_influence = False
        for i, well_pos in enumerate(self.well_positions):
            if self.wells_active[i]:
                vec_to_well = well_pos - asteroid['pos']
                dist_sq = vec_to_well.length_squared()
                
                if dist_sq < self.GRAVITY_WELL_RADIUS ** 2:
                    is_in_well_influence = True
                    if dist_sq > 1: # Avoid division by zero
                        # "Momentum increases pull strength"
                        speed = asteroid['vel'].length()
                        pull_strength = self.GRAVITY_STRENGTH * (1 + speed)
                        force_magnitude = pull_strength / dist_sq
                        total_force += vec_to_well.normalize() * force_magnitude

        # Update velocity and position
        asteroid['vel'] += total_force / self.FPS
        asteroid['pos'] += asteroid['vel']

        # Screen wrapping
        if asteroid['pos'].x < 0: asteroid['pos'].x += self.WIDTH
        if asteroid['pos'].x > self.WIDTH: asteroid['pos'].x -= self.WIDTH
        if asteroid['pos'].y < 0: asteroid['pos'].y += self.HEIGHT
        if asteroid['pos'].y > self.HEIGHT: asteroid['pos'].y -= self.HEIGHT

        # Handle transformation
        if is_in_well_influence and not asteroid['transformed']:
            asteroid['time_in_well'] += 1
            if asteroid['time_in_well'] >= self.TRANSFORM_TIME_STEPS:
                asteroid['transformed'] = True
                asteroid['radius'] = self.ASTEROID_RADIUS_INITIAL * 0.7 # Smaller
                asteroid['vel'] *= 1.5 # Faster
        elif not is_in_well_influence:
            asteroid['time_in_well'] = 0

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.current_asteroid_speed += self.ASTEROID_SPEED_INCREASE_RATE
            self.current_spawn_freq_hz = min(
                self.ASTEROID_SPAWN_FREQ_MAX_HZ,
                self.current_spawn_freq_hz + self.ASTEROID_SPAWN_FREQ_INCREASE_RATE
            )

    def _spawn_asteroids(self):
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            spawn_edge = self.np_random.integers(4)
            if spawn_edge == 0: # Top
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ASTEROID_RADIUS_INITIAL)
                angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
            elif spawn_edge == 1: # Bottom
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ASTEROID_RADIUS_INITIAL)
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            elif spawn_edge == 2: # Left
                pos = pygame.math.Vector2(-self.ASTEROID_RADIUS_INITIAL, self.np_random.uniform(0, self.HEIGHT))
                angle = self.np_random.uniform(-math.pi * 0.25, math.pi * 0.25)
            else: # Right
                pos = pygame.math.Vector2(self.WIDTH + self.ASTEROID_RADIUS_INITIAL, self.np_random.uniform(0, self.HEIGHT))
                angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)
            
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * self.current_asteroid_speed
            
            self.asteroids.append({
                'pos': pos,
                'vel': vel,
                'radius': self.ASTEROID_RADIUS_INITIAL,
                'transformed': False,
                'time_in_well': 0,
            })
            
            self.asteroid_spawn_timer = self.FPS / self.current_spawn_freq_hz

    def _create_capture_particles(self, pos, transformed):
        num_particles = 40 if transformed else 20
        color = self.COLOR_ASTEROID_TRANSFORMED if transformed else self.COLOR_ASTEROID
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30), # 0.5 to 1 second life
                'color': color,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background_elements()
        self._render_wells()
        self._render_particles()
        self._render_asteroids()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_elements(self):
        # Central capture point
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.CAPTURE_POINT_POS.x), int(self.CAPTURE_POINT_POS.y),
            self.CAPTURE_POINT_RADIUS + 10, self.COLOR_CAPTURE_GLOW
        )
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.CAPTURE_POINT_POS.x), int(self.CAPTURE_POINT_POS.y),
            self.CAPTURE_POINT_RADIUS, self.COLOR_CAPTURE_POINT
        )
        
        # Inactive well zones
        for pos in self.well_positions:
            pygame.gfxdraw.aacircle(
                self.screen, int(pos.x), int(pos.y),
                self.GRAVITY_WELL_RADIUS, self.COLOR_WELL_INACTIVE
            )

    def _render_wells(self):
        for i, pos in enumerate(self.well_positions):
            if self.wells_active[i]:
                # Pulsing effect for active well
                pulse_rad = self.GRAVITY_WELL_RADIUS + 10 * (1 + math.sin(self.well_activation_visual_timer * 0.5))
                alpha = 50 + 40 * (1 + math.sin(self.well_activation_visual_timer * 0.5))
                
                temp_surf = pygame.Surface((pulse_rad*2, pulse_rad*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(
                    temp_surf, int(pulse_rad), int(pulse_rad), int(pulse_rad),
                    (*self.COLOR_WELL_PULSE, int(alpha))
                )
                self.screen.blit(temp_surf, (int(pos.x - pulse_rad), int(pos.y - pulse_rad)))

                # Solid core for active well
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_WELL_ACTIVE)
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_WELL_ACTIVE)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pos_x, pos_y = int(asteroid['pos'].x), int(asteroid['pos'].y)
            radius = int(asteroid['radius'])
            color = self.COLOR_ASTEROID_TRANSFORMED if asteroid['transformed'] else self.COLOR_ASTEROID
            
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, color)

            # Transformation progress bar
            if not asteroid['transformed'] and asteroid['time_in_well'] > 0:
                progress = asteroid['time_in_well'] / self.TRANSFORM_TIME_STEPS
                bar_width = int(progress * radius * 2)
                bar_height = 3
                rect = pygame.Rect(pos_x - radius, pos_y + radius + 3, bar_width, bar_height)
                pygame.draw.rect(self.screen, self.COLOR_ASTEROID_TRANSFORMED, rect)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / 30.0
            alpha = int(255 * life_ratio)
            color = (*p['color'], alpha)
            size = int(3 * life_ratio)
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left_seconds = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font.render(f"TIME: {time_left_seconds:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        if self.game_over:
            end_text = self.font.render("TIME UP!", True, self.COLOR_WELL_ACTIVE)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS,
            "asteroids_on_screen": len(self.asteroids)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It will not be run during testing, so you can leave it as is.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    pygame.display.set_caption("Gravity Well")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        
        active_key_found = False
        for key, act_val in key_to_action.items():
            if keys[key] and not active_key_found:
                action[0] = act_val
                active_key_found = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        env.clock.tick(env.FPS)
        
    env.close()