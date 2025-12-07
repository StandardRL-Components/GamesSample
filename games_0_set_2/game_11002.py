import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:26:07.957453
# Source Brief: brief_01002.md
# Brief Index: 1002
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a particle.
    The goal is to change the particle's type by passing through colored gates
    and collect a specific number of each type at the top of the screen
    before time runs out.
    """
    metadata = {"render_modes": ["rgb_array", "human_playable"]}
    game_description = (
        "Control a particle, pass through colored gates to change its type, and collect the "
        "required number of each type at the top of the screen before time runs out."
    )
    user_guide = "Controls: Use ← and → arrow keys to move the particle left and right."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    COLOR_BG = (15, 15, 35)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_NEUTRAL = (255, 255, 255)
    TYPE_COLORS = {
        1: (255, 50, 50),   # Type A: Red
        2: (50, 255, 50),   # Type B: Green
        3: (50, 100, 255),  # Type C: Blue
    }
    TYPE_NAMES = {1: 'A', 2: 'B', 3: 'C'}

    GAME_DURATION_SECONDS = 45
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    WIN_CONDITION_COUNT = 5

    PARTICLE_RADIUS = 10
    PARTICLE_START_POS = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30]
    PARTICLE_HORIZONTAL_SPEED = 6
    PARTICLE_BASE_VERTICAL_SPEED = 2.5
    PARTICLE_MAX_VERTICAL_SPEED = 8
    PARTICLE_TRAIL_LENGTH = 15

    GATE_Y_POS = SCREEN_HEIGHT * 0.4
    GATE_HEIGHT = 10
    GATE_WIDTH = 100

    COLLECTION_Y_THRESHOLD = 25
    CONSECUTIVE_BONUS_THRESHOLD = 3
    SPEED_BOOST_MULTIPLIER = 1.15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.particle_pos = [0, 0]
        self.particle_type = 0
        self.particle_vertical_speed = 0
        self.particle_trail = deque(maxlen=self.PARTICLE_TRAIL_LENGTH)
        self.collected_counts = {1: 0, 2: 0, 3: 0}
        self.last_collected_type = 0
        self.consecutive_count = 0
        self.gates = []
        self._setup_gates()
        self.effects = {}
        self.stars = []
        self._setup_background_stars()

        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # this is for debugging, not needed in final code

    def _setup_gates(self):
        self.gates = []
        num_gates = len(self.TYPE_COLORS)
        gap = (self.SCREEN_WIDTH - num_gates * self.GATE_WIDTH) / (num_gates + 1)
        for i, type_id in enumerate(self.TYPE_COLORS.keys()):
            x_pos = gap * (i + 1) + self.GATE_WIDTH * i
            self.gates.append({
                'rect': pygame.Rect(x_pos, self.GATE_Y_POS, self.GATE_WIDTH, self.GATE_HEIGHT),
                'type': type_id,
                'color': self.TYPE_COLORS[type_id]
            })

    def _setup_background_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append({
                'pos': [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                'speed': random.uniform(0.1, 0.5),
                'size': random.randint(1, 2)
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.particle_pos = list(self.PARTICLE_START_POS)
        self.particle_type = 0
        self.particle_vertical_speed = self.PARTICLE_BASE_VERTICAL_SPEED
        self.particle_trail.clear()
        self.collected_counts = {1: 0, 2: 0, 3: 0}
        self.last_collected_type = 0
        self.consecutive_count = 0
        self.effects = {
            'explosions': [],
            'speed_lines': [],
            'screen_flash': {'color': None, 'alpha': 0}
        }

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.time_remaining -= 1 / self.FPS

        movement = action[0]
        if movement == 3:  # Left
            self.particle_pos[0] -= self.PARTICLE_HORIZONTAL_SPEED
        elif movement == 4:  # Right
            self.particle_pos[0] += self.PARTICLE_HORIZONTAL_SPEED

        self._update_particle()
        self._check_gate_passthrough()
        collection_reward = self._check_collection()
        reward += collection_reward
        self._update_effects()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            win = all(count >= self.WIN_CONDITION_COUNT for count in self.collected_counts.values())
            reward += 100 if win else -100
            # Sfx: Game Win / Game Lose

        self.score += reward

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _update_particle(self):
        self.particle_pos[1] -= self.particle_vertical_speed
        if self.particle_pos[0] < 0: self.particle_pos[0] = self.SCREEN_WIDTH
        elif self.particle_pos[0] > self.SCREEN_WIDTH: self.particle_pos[0] = 0
        self.particle_trail.append(list(self.particle_pos))

    def _check_gate_passthrough(self):
        particle_rect = pygame.Rect(self.particle_pos[0] - 2, self.particle_pos[1] - 2, 4, 4)
        for gate in self.gates:
            if gate['rect'].colliderect(particle_rect) and self.particle_type != gate['type']:
                self.particle_type = gate['type']
                # Sfx: Transform
                self.effects['screen_flash'] = {'color': gate['color'], 'alpha': 100}

    def _check_collection(self):
        reward = 0
        if self.particle_pos[1] <= self.COLLECTION_Y_THRESHOLD:
            if self.particle_type != 0:
                # Sfx: Collect
                self.collected_counts[self.particle_type] += 1
                reward += 0.1
                self._create_explosion(
                    (self._get_bin_x_pos(self.particle_type), self.COLLECTION_Y_THRESHOLD),
                    self.TYPE_COLORS[self.particle_type]
                )
                if self.particle_type == self.last_collected_type:
                    self.consecutive_count += 1
                else:
                    self.last_collected_type = self.particle_type
                    self.consecutive_count = 1

                if self.consecutive_count == self.CONSECUTIVE_BONUS_THRESHOLD:
                    # Sfx: Speed Boost
                    reward += 1.0
                    self.particle_vertical_speed = min(
                        self.PARTICLE_MAX_VERTICAL_SPEED,
                        self.particle_vertical_speed * self.SPEED_BOOST_MULTIPLIER
                    )
                    self.consecutive_count = 0
            
            self.particle_pos = list(self.PARTICLE_START_POS)
            self.particle_type = 0
            self.particle_trail.clear()
        return reward

    def _check_termination(self):
        win = all(count >= self.WIN_CONDITION_COUNT for count in self.collected_counts.values())
        timeout = self.time_remaining <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS
        return win or timeout or max_steps_reached

    def _update_effects(self):
        for explosion in self.effects['explosions']:
            for p in explosion:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['lifetime'] -= 1
        self.effects['explosions'] = [exp for exp in self.effects['explosions'] if any(p['lifetime'] > 0 for p in exp)]

        if self.particle_vertical_speed > self.PARTICLE_BASE_VERTICAL_SPEED + 0.1:
            if len(self.effects['speed_lines']) < 30:
                self.effects['speed_lines'].append([random.randint(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT])
        for line in self.effects['speed_lines']:
            line[1] -= self.particle_vertical_speed * 4
        self.effects['speed_lines'] = [line for line in self.effects['speed_lines'] if line[1] > 0]
        
        if self.effects['screen_flash']['alpha'] > 0:
            self.effects['screen_flash']['alpha'] = max(0, self.effects['screen_flash']['alpha'] - 20)

    def _get_observation(self):
        self._render_background()
        self._render_gates()
        self._render_particle()
        self._render_effects()
        self._render_ui()
        if self.game_over: self._render_game_over()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "collected_A": self.collected_counts[1],
            "collected_B": self.collected_counts[2],
            "collected_C": self.collected_counts[3],
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.SCREEN_HEIGHT:
                star['pos'] = [random.uniform(0, self.SCREEN_WIDTH), 0]
            pygame.draw.circle(self.screen, (100, 100, 120), star['pos'], star['size'])
            
    def _render_particle(self):
        color = self.TYPE_COLORS.get(self.particle_type, self.COLOR_NEUTRAL)
        pos_int = (int(self.particle_pos[0]), int(self.particle_pos[1]))
        
        for i, pos in enumerate(self.particle_trail):
            alpha = (i / self.PARTICLE_TRAIL_LENGTH) * 100
            self._draw_circle_alpha(self.screen, color + (int(alpha),), pos, self.PARTICLE_RADIUS * (i / self.PARTICLE_TRAIL_LENGTH))

        self._draw_circle_alpha(self.screen, color + (80,), pos_int, self.PARTICLE_RADIUS * 1.8)
        self._draw_circle_alpha(self.screen, color + (120,), pos_int, self.PARTICLE_RADIUS * 1.4)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PARTICLE_RADIUS, color)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PARTICLE_RADIUS, color)

    def _render_gates(self):
        for gate in self.gates:
            pulse = (math.sin(pygame.time.get_ticks() * 0.002 + gate['type']) + 1) / 2
            alpha = 30 + pulse * 40
            fill_color = gate['color'] + (int(alpha),)
            surf = pygame.Surface(gate['rect'].size, pygame.SRCALPHA)
            pygame.draw.rect(surf, fill_color, surf.get_rect())
            self.screen.blit(surf, gate['rect'].topleft)
            pygame.draw.rect(self.screen, gate['color'], gate['rect'], 2)

    def _render_effects(self):
        for explosion in self.effects['explosions']:
            for p in explosion:
                if p['lifetime'] > 0:
                    alpha = max(0, min(255, p['lifetime'] * 10))
                    self._draw_circle_alpha(self.screen, p['color'] + (int(alpha),), p['pos'], p['size'])
        
        for line_pos in self.effects['speed_lines']:
            start_pos = (int(line_pos[0]), int(line_pos[1]))
            end_pos = (int(line_pos[0]), int(line_pos[1] + self.particle_vertical_speed * 2))
            pygame.draw.line(self.screen, (200, 200, 255), start_pos, end_pos, 1)

        if self.effects['screen_flash']['alpha'] > 0:
            flash_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surf.fill(self.effects['screen_flash']['color'] + (int(self.effects['screen_flash']['alpha']),))
            self.screen.blit(flash_surf, (0, 0))

    def _render_ui(self):
        for type_id, name in self.TYPE_NAMES.items():
            x_pos = self._get_bin_x_pos(type_id)
            color = self.TYPE_COLORS[type_id]
            count = self.collected_counts[type_id]
            pygame.draw.rect(self.screen, color, (x_pos - 25, 0, 50, 10))
            text_surf = self.font_small.render(f"{name}: {count}/{self.WIN_CONDITION_COUNT}", True, self.COLOR_UI_TEXT)
            self.screen.blit(text_surf, (x_pos - text_surf.get_width() // 2, 15))

        time_surf = self.font_small.render(f"TIME: {max(0, math.ceil(self.time_remaining)):02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH // 2 - time_surf.get_width() // 2, 15))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        win = all(count >= self.WIN_CONDITION_COUNT for count in self.collected_counts.values())
        message, color = ("VICTORY!", (100, 255, 100)) if win else ("TIME UP", (255, 100, 100))
        text_surf = self.font_large.render(message, True, color)
        self.screen.blit(text_surf, text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _get_bin_x_pos(self, type_id):
        return self.SCREEN_WIDTH * type_id / 4

    def _create_explosion(self, pos, color, num_particles=20):
        explosion = []
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            explosion.append({'pos': list(pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                              'lifetime': random.randint(20, 40), 'size': random.uniform(1, 4), 'color': color})
        self.effects['explosions'].append(explosion)

    def _draw_circle_alpha(self, surface, color, center, radius):
        if radius <= 0: return
        target_rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (int(radius), int(radius)), int(radius))
        surface.blit(shape_surf, target_rect)
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="human_playable")
    obs, info = env.reset()
    
    # This check is useful for debugging, but not required for the final submission.
    # env.validate_implementation()
    
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Particle Collector")

    done = False
    total_reward = 0
    
    while not done:
        keys = pygame.key.get_pressed()
        movement_action = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement_action = 4
        action = [movement_action, 0, 0]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()