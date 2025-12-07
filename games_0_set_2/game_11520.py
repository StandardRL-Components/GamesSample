import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:06:08.773321
# Source Brief: brief_01520.md
# Brief Index: 1520
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a chaotic field of obstacles by flipping gravity. "
        "Trigger chain reactions to clear your path and reach the finish line before time runs out."
    )
    user_guide = (
        "Controls: Press space to flip gravity. Avoid red obstacles and hit blue boosts to reach the finish line."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    
    # Colors
    COLOR_BG = (10, 5, 15)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_OBSTACLE = (255, 60, 60)
    COLOR_BOOST = (60, 150, 255)
    COLOR_FIELD = (180, 0, 255)
    COLOR_FINISH = (255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PROGRESS_BAR_BG = (50, 50, 70)
    COLOR_PROGRESS_BAR_FG = (0, 255, 150)

    # Physics
    GRAVITY_STRENGTH = 0.15
    PLAYER_DAMPING = 0.995
    PLAYER_MAX_SPEED = 10
    BOOST_STRENGTH = 1.5
    BOOST_DURATION = 90  # frames
    OBSTACLE_COLLISION_DAMPING = 0.5
    FIELD_STRENGTH = 0.002
    
    # Game Rules
    MAX_EPISODE_STEPS = 1800 # 30 seconds at 60 FPS
    INITIAL_TIME_LIMIT = 60.0 # seconds
    INITIAL_OBSTACLE_COUNT = 15
    FINISH_LINE_X = SCREEN_WIDTH - 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_timer = pygame.font.SysFont('Consolas', 30, bold=True)

        # Game state that persists across resets for difficulty scaling
        self.current_time_limit = self.INITIAL_TIME_LIMIT
        self.current_obstacle_count = self.INITIAL_OBSTACLE_COUNT
        
        # Initialize state variables
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.gravity_direction = 1 # 1 for down, -1 for up
        self.time_remaining = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_state = False
        self.boost_timer = 0
        self.last_progress = 0.0

        self.obstacles = []
        self.speed_boosts = []
        self.magnetic_fields = []
        self.particles = []
        self.player_trail = []

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)
        
        # Adjust difficulty if last run was a win
        if hasattr(self, 'game_won') and self.game_won:
            self.current_time_limit = max(15.0, self.current_time_limit - 5.0)
            self.current_obstacle_count = min(50, int(self.current_obstacle_count * 1.05))
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = self.current_time_limit

        self.player_pos = pygame.math.Vector2(30, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(2, 0)
        self.gravity_direction = 1
        self.last_space_state = False
        self.boost_timer = 0
        self.last_progress = self.player_pos.x / self.FINISH_LINE_X

        self.particles.clear()
        self.player_trail.clear()
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.obstacles = []
        self.speed_boosts = []
        self.magnetic_fields = []

        for _ in range(self.current_obstacle_count):
            pos = pygame.math.Vector2(
                random.uniform(100, self.SCREEN_WIDTH - 100),
                random.uniform(20, self.SCREEN_HEIGHT - 20)
            )
            self.obstacles.append({'pos': pos, 'radius': random.randint(8, 15), 'triggered': False})

        for _ in range(5):
            pos = pygame.math.Vector2(
                random.uniform(100, self.SCREEN_WIDTH - 100),
                random.uniform(20, self.SCREEN_HEIGHT - 20)
            )
            self.speed_boosts.append({'pos': pos, 'radius': 10})

        for _ in range(3):
            pos = pygame.math.Vector2(
                random.uniform(100, self.SCREEN_WIDTH - 100),
                random.uniform(20, self.SCREEN_HEIGHT - 20)
            )
            strength = random.uniform(0.8, 1.2)
            self.magnetic_fields.append({'pos': pos, 'radius': random.randint(50, 80), 'strength': strength})

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        # --- Action Handling ---
        # movement = action[0] # Unused per brief
        space_pressed = action[1] == 1
        # shift_held = action[2] == 1 # Unused per brief

        if space_pressed and not self.last_space_state:
            self.gravity_direction *= -1
            # sfx: gravity_flip.wav
        self.last_space_state = space_pressed

        # --- Physics and Game Logic Update ---
        self._update_player_physics()
        self._update_particles()
        reward += self._handle_collisions()
        
        # --- Progress Reward ---
        current_progress = self.player_pos.x / self.FINISH_LINE_X
        progress_delta = current_progress - self.last_progress
        if progress_delta > 0:
            reward += progress_delta * 10
        self.last_progress = current_progress

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.time_remaining <= 0:
            terminated = True
            self.game_over = True
            reward -= 100 # Time out penalty
            # sfx: game_lose.wav
        
        if self.player_pos.x >= self.FINISH_LINE_X:
            terminated = True
            self.game_over = True
            self.game_won = True
            reward += 100 # Win bonus
            # sfx: game_win.wav

        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True
            # No specific reward/penalty, timeout is the primary loss condition
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player_physics(self):
        # Apply gravity
        acceleration = pygame.math.Vector2(0, self.GRAVITY_STRENGTH * self.gravity_direction)
        
        # Apply magnetic fields
        for field in self.magnetic_fields:
            vec_to_player = self.player_pos - field['pos']
            dist_sq = vec_to_player.length_squared()
            if 0 < dist_sq < field['radius']**2:
                force_mag = 1 - (math.sqrt(dist_sq) / field['radius'])
                force = vec_to_player.normalize() * force_mag * field['strength']
                acceleration += force

        # Update velocity
        self.player_vel += acceleration
        
        # Apply speed boost
        if self.boost_timer > 0:
            self.boost_timer -= 1
            boost_factor = self.BOOST_STRENGTH
        else:
            boost_factor = 1.0

        # Apply damping and clamp speed
        self.player_vel *= self.PLAYER_DAMPING
        if self.player_vel.length() > self.PLAYER_MAX_SPEED * boost_factor:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED * boost_factor)

        # Update position and handle toroidal world wrap
        self.player_pos += self.player_vel
        self.player_pos.x %= self.SCREEN_WIDTH
        self.player_pos.y %= self.SCREEN_HEIGHT

        # Update player trail
        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > 20:
            self.player_trail.pop(0)

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Obstacles
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 5, 10, 10)
        for obs in self.obstacles:
            if not obs['triggered']:
                dist = self.player_pos.distance_to(obs['pos'])
                if dist < 10 + obs['radius']:
                    obs['triggered'] = True
                    self.player_vel *= self.OBSTACLE_COLLISION_DAMPING
                    reward -= 0.5 # Collision penalty
                    reward += 2.0 # Chain reaction trigger bonus
                    self._create_explosion(obs['pos'], self.COLOR_OBSTACLE, 20)
                    # sfx: obstacle_hit.wav
        
        # Player vs Speed Boosts
        for boost in self.speed_boosts[:]:
            dist = self.player_pos.distance_to(boost['pos'])
            if dist < 10 + boost['radius']:
                self.speed_boosts.remove(boost)
                self.boost_timer = self.BOOST_DURATION
                reward += 1.0 # Boost collection bonus
                self._create_explosion(boost['pos'], self.COLOR_BOOST, 15, is_boost=True)
                # sfx: boost_collect.wav
        
        # Cleanup triggered obstacles for chain reaction propagation
        self.obstacles = [obs for obs in self.obstacles if not obs.get('destroy_me', False)]

        return reward

    def _create_explosion(self, pos, color, count, is_boost=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4) if not is_boost else random.uniform(0.5, 2)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(20, 40)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color, 'is_boost': is_boost})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if not p['is_boost']:
                p['vel'] *= 0.95 # Air resistance for explosion particles
            
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                continue
            
            # Chain reaction check
            if not p['is_boost']:
                for obs in self.obstacles:
                    if not obs['triggered']:
                        if obs['pos'].distance_to(p['pos']) < obs['radius']:
                            obs['triggered'] = True
                            # No player reward for secondary explosions
                            self._create_explosion(obs['pos'], self.COLOR_OBSTACLE, 20)
                            # sfx: obstacle_hit_secondary.wav

        # Mark triggered obstacles for removal after a short delay
        for obs in self.obstacles:
            if obs['triggered']:
                obs.setdefault('trigger_time', 0)
                obs['trigger_time'] += 1
                if obs['trigger_time'] > 5: # 5 frames delay
                    obs['destroy_me'] = True


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw magnetic fields (pulsating aura)
        pulse = math.sin(self.steps * 0.05) * 5
        for field in self.magnetic_fields:
            radius = int(field['radius'] + pulse)
            if radius > 0:
                self._draw_glow_circle(field['pos'], radius, self.COLOR_FIELD, 0.1)
        
        # Draw finish line
        finish_color_pulse = int(128 + 127 * math.sin(self.steps * 0.1))
        finish_color = (finish_color_pulse, finish_color_pulse, 100)
        pygame.draw.line(self.screen, finish_color, (self.FINISH_LINE_X, 0), (self.FINISH_LINE_X, self.SCREEN_HEIGHT), 3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40.0))
            color = p['color']
            size = int(max(1, 5 * (p['lifespan'] / 40.0))) if not p['is_boost'] else 3
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (color[0], color[1], color[2], alpha), (size, size), size)
            self.screen.blit(s, (int(p['pos'].x - size), int(p['pos'].y - size)))

        # Draw player trail
        if len(self.player_trail) > 1:
            for i in range(1, len(self.player_trail)):
                alpha = int(100 * (i / len(self.player_trail)))
                color = self.COLOR_PLAYER
                start_pos = self.player_trail[i-1]
                end_pos = self.player_trail[i]
                pygame.draw.line(self.screen, (color[0], color[1], color[2], alpha), start_pos, end_pos, 3)

        # Draw speed boosts
        for boost in self.speed_boosts:
            self._draw_glow_circle(boost['pos'], boost['radius'], self.COLOR_BOOST, 0.5)

        # Draw obstacles
        for obs in self.obstacles:
            if not obs.get('destroy_me', False):
                self._draw_glow_circle(obs['pos'], obs['radius'], self.COLOR_OBSTACLE, 0.6)

        # Draw player
        player_color = self.COLOR_BOOST if self.boost_timer > 0 else self.COLOR_PLAYER
        self._draw_glow_circle(self.player_pos, 10, player_color, 1.0)
        
        # Draw gravity indicator
        indicator_y = 15 if self.gravity_direction == -1 else self.SCREEN_HEIGHT - 15
        indicator_color = (255, 255, 255, 100)
        start_pos = (int(self.player_pos.x - 10), indicator_y)
        end_pos = (int(self.player_pos.x + 10), indicator_y)
        pygame.draw.line(self.screen, indicator_color, start_pos, end_pos, 2)


    def _draw_glow_circle(self, pos, radius, color, intensity):
        x, y = int(pos.x), int(pos.y)
        
        # Draw outer glow layers
        for i in range(int(radius * 1.5), int(radius), -2):
            alpha = int(intensity * 50 * (1 - (i - radius) / (radius * 0.5)))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, x, y, i, (color[0], color[1], color[2], alpha))

        # Draw main circle
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius), color)

    def _render_ui(self):
        # Timer
        time_str = f"{self.time_remaining:.2f}"
        timer_surf = self.font_timer.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 5))

        # Score
        score_str = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 5))

        # Progress bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 10
        
        progress = min(1.0, max(0.0, self.player_pos.x / self.FINISH_LINE_X))
        
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_FG, (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "progress": self.last_progress
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    terminated, truncated = False, False
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Boson Blitz - Manual Control")

    total_reward = 0
    
    # Action state
    action = [0, 0, 0] # [movement, space, shift]

    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the step function to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Time: {info['time_remaining']:.2f}")
            # Reset for another round
            obs, info = env.reset()
            terminated, truncated = False, False
            action = [0, 0, 0]

    env.close()