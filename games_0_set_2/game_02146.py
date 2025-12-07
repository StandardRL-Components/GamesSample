
# Generated: 2025-08-27T19:23:46.469570
# Source Brief: brief_02146.md
# Brief Index: 2146

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, color, initial_radius, final_radius, lifetime):
        self.pos = list(pos)
        self.color = color
        self.initial_radius = initial_radius
        self.final_radius = final_radius
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]

    def update(self):
        self.lifetime -= 1
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.velocity[0] *= 0.95  # Dampen velocity
        self.velocity[1] *= 0.95

    def draw(self, surface):
        if self.lifetime > 0:
            progress = self.lifetime / self.max_lifetime
            current_radius = int(self.initial_radius + (self.final_radius - self.initial_radius) * (1 - progress))
            alpha = int(255 * progress)
            
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.color + (alpha,), (current_radius, current_radius), current_radius)
            surface.blit(temp_surf, (int(self.pos[0] - current_radius), int(self.pos[1] - current_radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your circle. Collect orbs that match your color."
    )

    game_description = (
        "Navigate a vibrant arena to collect color-matched orbs before time runs out. "
        "Your color changes periodically, so adapt your strategy quickly!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.DELTA_TIME = 1.0 / self.FPS
        
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_sub = pygame.font.SysFont("Consolas", 20)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_BOUNDARY = (200, 200, 220)
        self.COLOR_PLAYER = (255, 255, 255)
        self.ORB_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
        ]
        self.COLOR_UI = (240, 240, 240)
        
        # Player state
        self.player_pos = np.array([0.0, 0.0])
        self.player_radius = 12
        self.player_speed = 150  # pixels per second

        # Game state
        self.orbs = []
        self.particles = []
        self.max_orbs = 15
        self.orb_radius = 10
        self.player_color_index = 0
        
        self.max_time = 180.0
        self.player_color_change_interval = 10.0 # seconds
        self.orb_color_shift_interval = 5.0 # seconds
        self.win_condition_orbs = 50
        self.max_episode_steps = 6000 # ~200s at 30fps

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # For internal testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.orbs_collected = 0
        
        self.timer = self.max_time
        self.player_color_change_timer = self.player_color_change_interval
        self.orb_color_shift_timer = self.orb_color_shift_interval
        
        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        self.player_color_index = self.np_random.integers(0, len(self.ORB_COLORS))
        
        self.orbs.clear()
        self.particles.clear()
        for _ in range(self.max_orbs):
            self._spawn_orb()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # 1. Calculate continuous reward for movement
        reward += self._calculate_continuous_reward(action)
        
        # 2. Update game logic
        self._update_player(action)
        self._update_timers()
        
        # 3. Handle collisions and event-based rewards
        event_reward = self._handle_collisions()
        reward += event_reward
        
        # 4. Update environment
        self._update_particles()
        self._cull_and_respawn_orbs()
        
        self.steps += 1
        
        # 5. Check termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.orbs_collected >= self.win_condition_orbs:
                reward += 100 # Win bonus
            elif self.timer <= 0:
                reward -= 100 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_continuous_reward(self, action):
        if action[0] == 0: # no-op
            return 0

        matching_orbs = [orb for orb in self.orbs if orb['color_index'] == self.player_color_index]
        if not matching_orbs:
            return 0

        # Find closest matching orb
        closest_orb = min(matching_orbs, key=lambda o: np.linalg.norm(self.player_pos - o['pos']))
        dist_before = np.linalg.norm(self.player_pos - closest_orb['pos'])
        
        # Simulate next position
        vel = self._get_velocity_from_action(action)
        next_pos = self.player_pos + vel * self.DELTA_TIME
        dist_after = np.linalg.norm(next_pos - closest_orb['pos'])
        
        if dist_after < dist_before:
            return 1.0 # Moving towards correct orb
        else:
            return -0.1 # Moving away from correct orb

    def _get_velocity_from_action(self, action):
        movement = action[0]
        vel = np.array([0.0, 0.0])
        if movement == 1: vel[1] -= self.player_speed
        elif movement == 2: vel[1] += self.player_speed
        elif movement == 3: vel[0] -= self.player_speed
        elif movement == 4: vel[0] += self.player_speed
        return vel

    def _update_player(self, action):
        vel = self._get_velocity_from_action(action)
        self.player_pos += vel * self.DELTA_TIME
        
        # Clamp position to stay within arena boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.WIDTH - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.HEIGHT - self.player_radius)

    def _update_timers(self):
        self.timer = max(0, self.timer - self.DELTA_TIME)
        
        self.player_color_change_timer -= self.DELTA_TIME
        if self.player_color_change_timer <= 0:
            self.player_color_index = self.np_random.integers(0, len(self.ORB_COLORS))
            self.player_color_change_timer = self.player_color_change_interval
            # sfx: player color change chime
        
        self.orb_color_shift_timer -= self.DELTA_TIME
        if self.orb_color_shift_timer <= 0:
            for orb in self.orbs:
                orb['color_index'] = (orb['color_index'] + 1) % len(self.ORB_COLORS)
            self.orb_color_shift_timer = self.orb_color_shift_interval
            # sfx: orb color shift sound

    def _handle_collisions(self):
        reward = 0
        orbs_to_remove = []
        for orb in self.orbs:
            dist = np.linalg.norm(self.player_pos - orb['pos'])
            if dist < self.player_radius + self.orb_radius:
                orbs_to_remove.append(orb)
                if orb['color_index'] == self.player_color_index:
                    self.score += 10
                    reward += 10
                    self.orbs_collected += 1
                    # sfx: positive collection sound
                    self._create_particle_effect(orb['pos'], self.ORB_COLORS[orb['color_index']], 20)
                else:
                    self.score -= 5
                    reward -= 5
                    # sfx: negative collection sound
                    self._create_particle_effect(orb['pos'], (128,128,128), 10)

        self.orbs = [orb for orb in self.orbs if orb not in orbs_to_remove]
        return reward

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _cull_and_respawn_orbs(self):
        while len(self.orbs) < self.max_orbs:
            self._spawn_orb()

    def _check_termination(self):
        return (self.timer <= 0 or 
                self.orbs_collected >= self.win_condition_orbs or 
                self.steps >= self.max_episode_steps)

    def _spawn_orb(self):
        # Ensure orb doesn't spawn too close to the player
        while True:
            pos = np.array([
                self.np_random.uniform(self.orb_radius, self.WIDTH - self.orb_radius),
                self.np_random.uniform(self.orb_radius, self.HEIGHT - self.orb_radius)
            ])
            if np.linalg.norm(pos - self.player_pos) > self.player_radius + self.orb_radius + 20:
                break
        
        color_index = self.np_random.integers(0, len(self.ORB_COLORS))
        self.orbs.append({'pos': pos, 'color_index': color_index})

    def _create_particle_effect(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos, color, 2, 15, 20))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw orbs
        for orb in self.orbs:
            pos_int = (int(orb['pos'][0]), int(orb['pos'][1]))
            color = self.ORB_COLORS[orb['color_index']]
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.orb_radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.orb_radius, color)

        # Draw player
        player_pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        player_color = self.ORB_COLORS[self.player_color_index]
        
        # Player glow
        glow_radius = int(self.player_radius * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, player_color + (80,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_pos_int[0] - glow_radius, player_pos_int[1] - glow_radius))
        
        # Player core
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        timer_text = self.font_main.render(f"TIME: {math.ceil(self.timer)}", True, self.COLOR_UI)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)
        
        # Orbs collected
        orbs_text = self.font_sub.render(f"ORBS: {self.orbs_collected} / {self.win_condition_orbs}", True, self.COLOR_UI)
        orbs_rect = orbs_text.get_rect(midbottom=(self.WIDTH / 2, self.HEIGHT - 10))
        self.screen.blit(orbs_text, orbs_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orbs_collected": self.orbs_collected,
            "timer": self.timer,
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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