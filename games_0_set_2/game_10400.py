import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:20:42.478273
# Source Brief: brief_00400.md
# Brief Index: 400
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent guides splitting particles to target locations.
    The goal is to achieve 80% proximity for three tiers of particles (Large, Medium, Small)
    to their respective targets within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide groups of particles to their targets by applying directional nudges. "
        "Collide particles to split them into smaller tiers and solve the puzzle before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to nudge the selected particle group. "
        "Use space and shift to cycle between particle tiers (Large, Medium, Small)."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    # The brief implies a 30FPS simulation for a 90s game. 90 * 30 = 2700.
    MAX_STEPS = 2700
    WIN_PROXIMITY = 0.8
    NUDGE_STRENGTH = 0.15
    PARTICLE_DRAG = 0.995
    MAX_VELOCITY = 3.0

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_RED = (255, 60, 60)
    COLOR_GREEN = (60, 255, 60)
    COLOR_BLUE = (60, 120, 255)
    COLOR_WHITE = (230, 230, 230)
    TIER_COLORS = [COLOR_BLUE, COLOR_GREEN, COLOR_RED]  # Small, Medium, Large
    TIER_NAMES = ["SMALL", "MEDIUM", "LARGE"]

    # --- Tiers & Sizes ---
    TIER_SMALL, TIER_MEDIUM, TIER_LARGE = 0, 1, 2
    TIER_SIZES = {
        TIER_SMALL: 6,
        TIER_MEDIUM: 10,
        TIER_LARGE: 15
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_tier = pygame.font.Font(None, 32)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_tier = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.tier_reward_given = []
        self.targets = {}
        self.particles = []
        self.next_particle_id = 0
        self.stars = []
        
        # Note: reset() is called by the environment wrapper, no need to call it here.
        # self.reset()
        
        # self.validate_implementation() # This is a test helper, not part of the env logic.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.selected_tier = self.TIER_LARGE
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.tier_reward_given = [False, False, False]
        
        self._initialize_targets()
        self._initialize_particles()
        self._initialize_background()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            info = self._get_info()
            # The game is over, but we need to return a valid step tuple.
            # Terminated is True, so reward is conventionally 0.
            return obs, 0, True, False, info

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        
        self._handle_actions(space_held, shift_held)
        self._update_physics(movement)
        
        proximities = self._calculate_proximities()
        reward = self._calculate_reward(proximities)
        
        terminated = self._check_termination(proximities)
        truncated = self.steps >= self.MAX_STEPS # Truncated if timeout, terminated if win
        
        if terminated:
            self.game_over = True
            win = all(p >= self.WIN_PROXIMITY for p in proximities.values())
            terminal_reward = 100 if win else 0 # Reward for winning is given here
            reward += terminal_reward
            self.score += terminal_reward
        
        if truncated and not terminated:
             self.game_over = True
             reward += -100 # Penalty for timeout
             self.score += -100

        self.score += reward
        
        # Per Gymnasium API, step returns (obs, reward, terminated, truncated, info)
        # In this game, win is termination, timeout is truncation.
        is_terminated = self.game_over and not (self.steps >= self.MAX_STEPS)
        is_truncated = self.game_over and (self.steps >= self.MAX_STEPS) and not is_terminated


        return self._get_observation(), reward, is_terminated, is_truncated, self._get_info()

    def _handle_actions(self, space_held, shift_held):
        if space_held and not self.prev_space_held:
            # sound_placeholder: "UI_SWITCH_UP.WAV"
            self.selected_tier = (self.selected_tier + 1) % 3

        if shift_held and not self.prev_shift_held:
            # sound_placeholder: "UI_SWITCH_DOWN.WAV"
            self.selected_tier = (self.selected_tier - 1 + 3) % 3
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_physics(self, movement_action):
        nudge = pygame.Vector2(0, 0)
        if movement_action == 1: nudge.y = -self.NUDGE_STRENGTH
        elif movement_action == 2: nudge.y = self.NUDGE_STRENGTH
        elif movement_action == 3: nudge.x = -self.NUDGE_STRENGTH
        elif movement_action == 4: nudge.x = self.NUDGE_STRENGTH

        for p in self.particles:
            if p['tier'] == self.selected_tier:
                p['vel'] += nudge
                if p['vel'].length() > self.MAX_VELOCITY:
                    p['vel'].scale_to_length(self.MAX_VELOCITY)
            
            p['vel'] *= self.PARTICLE_DRAG
            p['trail'].append(p['pos'].copy())
            p['pos'] += p['vel']
            
            if not (p['size'] < p['pos'].x < self.SCREEN_WIDTH - p['size']):
                p['vel'].x *= -1
                p['pos'].x = np.clip(p['pos'].x, p['size'], self.SCREEN_WIDTH - p['size'])
            if not (p['size'] < p['pos'].y < self.SCREEN_HEIGHT - p['size']):
                p['vel'].y *= -1
                p['pos'].y = np.clip(p['pos'].y, p['size'], self.SCREEN_HEIGHT - p['size'])
        
        self._handle_collisions()

    def _handle_collisions(self):
        collided_pairs = set()
        particles_to_create = []
        particles_to_remove = set()

        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                pair_id = tuple(sorted((p1['id'], p2['id'])))
                if pair_id in collided_pairs:
                    continue
                
                dist_vec = p1['pos'] - p2['pos']
                if not dist_vec.length_squared(): continue
                dist_len = dist_vec.length()

                if 0 < dist_len < (p1['size'] + p2['size']):
                    collided_pairs.add(pair_id)
                    # sound_placeholder: "COLLIDE.WAV"

                    # Elastic collision physics
                    v1, v2 = p1['vel'], p2['vel']
                    x1, x2 = p1['pos'], p2['pos']
                    m1, m2 = p1['size']**2, p2['size']**2
                    
                    if m1 + m2 == 0: continue

                    v1_new = v1 - (2 * m2 / (m1 + m2)) * (v1 - v2).dot(x1 - x2) / dist_len**2 * (x1 - x2)
                    v2_new = v2 - (2 * m1 / (m1 + m2)) * (v2 - v1).dot(x2 - x1) / dist_len**2 * (x2 - x1)
                    p1['vel'], p2['vel'] = v1_new, v2_new

                    splitter = p1 if p1['tier'] >= p2['tier'] else p2
                    if splitter['tier'] > self.TIER_SMALL:
                        if splitter['id'] not in particles_to_remove:
                            particles_to_remove.add(splitter['id'])
                            next_tier = splitter['tier'] - 1
                            for _ in range(2):
                                offset = pygame.Vector2(self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5))
                                new_vel = splitter['vel'].rotate(self.np_random.uniform(-45, 45)) * 1.1
                                particles_to_create.append({'tier': next_tier, 'pos': splitter['pos'] + offset, 'vel': new_vel})

        if particles_to_remove:
            self.particles = [p for p in self.particles if p['id'] not in particles_to_remove]
            for p_data in particles_to_create:
                self._create_particle(p_data['tier'], pos=p_data['pos'], vel=p_data['vel'])

    def _calculate_proximities(self):
        proximities = {}
        max_dist = math.hypot(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        for tier in [self.TIER_LARGE, self.TIER_MEDIUM, self.TIER_SMALL]:
            tier_particles = [p for p in self.particles if p['tier'] == tier]
            if not tier_particles:
                proximities[tier] = 0.0
                continue
            
            sum_x = sum(p['pos'].x for p in tier_particles)
            sum_y = sum(p['pos'].y for p in tier_particles)
            barycenter = pygame.Vector2(sum_x / len(tier_particles), sum_y / len(tier_particles))
            
            distance = barycenter.distance_to(self.targets[tier])
            proximity = max(0.0, 1.0 - (distance / max_dist))
            proximities[tier] = proximity
        
        return proximities

    def _calculate_reward(self, proximities):
        reward = 0.0
        prox_radius = self.SCREEN_WIDTH * 0.25
        for p in self.particles:
            target_pos = self.targets[p['tier']]
            if p['pos'].distance_to(target_pos) < prox_radius:
                reward += 0.01

        for tier, prox in proximities.items():
            if prox >= self.WIN_PROXIMITY and not self.tier_reward_given[tier]:
                reward += 1.0
                # sound_placeholder: "TIER_SYNC.WAV"
                self.tier_reward_given[tier] = True
        
        return reward

    def _check_termination(self, proximities):
        win = all(prox >= self.WIN_PROXIMITY for prox in proximities.values())
        if win:
            # sound_placeholder: "WIN_JINGLE.WAV"
            return True
        
        timeout = self.steps >= self.MAX_STEPS
        if timeout:
            # sound_placeholder: "LOSE_SOUND.WAV"
            return True # Game ends on timeout as well
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, b in self.stars:
            c = int(40 + 20 * math.sin(self.steps * 0.05 + x))
            color = (max(20, c), max(20, c), max(20, c + 10))
            pygame.draw.circle(self.screen, color, (x, y), b / 2)

    def _render_game_elements(self):
        barycenters = self._get_barycenters()

        for tier, pos in self.targets.items():
            color = self.TIER_COLORS[tier]
            if tier in barycenters:
                pygame.draw.line(self.screen, color + (50,), pos, barycenters[tier], 1)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 20, color + (100,))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 20, color + (50,))

        for p in sorted(self.particles, key=lambda x: x['size']):
            color = self.TIER_COLORS[p['tier']]
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            
            for i, trail_pos in enumerate(p['trail']):
                alpha = int(80 * (i / len(p['trail'])))
                trail_color = color + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(trail_pos.x), int(trail_pos.y), int(p['size'] * (i / len(p['trail'])) * 0.5), trail_color)

            if p['tier'] == self.selected_tier:
                for i in range(4):
                    glow_size = int(p['size'] * (1.2 + i * 0.15))
                    glow_alpha = 60 - i * 15
                    pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], glow_size, color + (glow_alpha,))

            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(p['size']), color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['size']), color)

    def _render_ui(self):
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 85), pygame.SRCALPHA)
        ui_panel.fill((20, 30, 50, 180))
        self.screen.blit(ui_panel, (0, 0))

        time_left = (self.MAX_STEPS - self.steps) / 30.0
        time_text = f"TIME: {max(0, time_left):.1f}s"
        text_surf = self.font_ui.render(time_text, True, self.COLOR_WHITE)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 15, 10))
        
        proximities = self._calculate_proximities()
        for i, tier in enumerate([self.TIER_LARGE, self.TIER_MEDIUM, self.TIER_SMALL]):
            prox_val = proximities.get(tier, 0.0)
            color = self.TIER_COLORS[tier]
            text = f"{self.TIER_NAMES[tier]}: {prox_val:.0%}"
            text_surf = self.font_ui.render(text, True, color)
            self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 15, 35 + i * 18))
        
        selected_text = f"SELECTED: {self.TIER_NAMES[self.selected_tier]}"
        color = self.TIER_COLORS[self.selected_tier]
        text_surf = self.font_tier.render(selected_text, True, color)
        self.screen.blit(text_surf, (15, 15))
        
        controls_text = "MOVE: ARROWS | CYCLE: SPACE/SHIFT"
        text_surf = self.font_ui.render(controls_text, True, self.COLOR_WHITE)
        self.screen.blit(text_surf, (15, 55))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _initialize_targets(self):
        self.targets[self.TIER_LARGE] = pygame.Vector2(100, 100)
        self.targets[self.TIER_MEDIUM] = pygame.Vector2(self.SCREEN_WIDTH - 100, 100)
        self.targets[self.TIER_SMALL] = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 100)
        
    def _initialize_particles(self):
        self.particles.clear()
        self.next_particle_id = 0
        for tier, num in zip([self.TIER_LARGE, self.TIER_MEDIUM, self.TIER_SMALL], [3, 4, 5]):
            for _ in range(num):
                pos = pygame.Vector2(
                    self.np_random.integers(low=self.SCREEN_WIDTH * 0.25, high=self.SCREEN_WIDTH * 0.75),
                    self.np_random.integers(low=self.SCREEN_HEIGHT * 0.25, high=self.SCREEN_HEIGHT * 0.75)
                )
                self._create_particle(tier, pos=pos)

    def _create_particle(self, tier, pos=None, vel=None):
        if pos is None:
            pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        if vel is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, 1.0)
        
        particle = {
            'id': self.next_particle_id,
            'pos': pos, 'vel': vel, 'tier': tier,
            'size': self.TIER_SIZES[tier],
            'trail': deque(maxlen=10)
        }
        self.particles.append(particle)
        self.next_particle_id += 1

    def _initialize_background(self):
        self.stars.clear()
        for _ in range(150):
            self.stars.append((
                self.np_random.integers(0, self.SCREEN_WIDTH),
                self.np_random.integers(0, self.SCREEN_HEIGHT),
                self.np_random.integers(1, 4)
            ))

    def _get_barycenters(self):
        barycenters = {}
        for tier in [self.TIER_LARGE, self.TIER_MEDIUM, self.TIER_SMALL]:
            tier_particles = [p for p in self.particles if p['tier'] == tier]
            if tier_particles:
                sum_x = sum(p['pos'].x for p in tier_particles)
                sum_y = sum(p['pos'].y for p in tier_particles)
                barycenters[tier] = pygame.Vector2(sum_x / len(tier_particles), sum_y / len(tier_particles))
        return barycenters

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # This method is for testing during development and can be removed or ignored.
        print("Running validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")