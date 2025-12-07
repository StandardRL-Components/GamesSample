import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" for headless execution, as required.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control three pendulums to catch falling orbs. Time your swings to collect all the orbs "
        "without letting them drop or letting the pendulums collide."
    )
    user_guide = (
        "Use ←, ↓, and → to select a pendulum. Press space to swing it left and shift to swing it right. "
        "Catch the orbs before they fall!"
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    ORBS_TO_WIN = 10

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (40, 60, 100)
    COLOR_PENDULUM_1 = (255, 80, 80)
    COLOR_PENDULUM_2 = (80, 255, 80)
    COLOR_PENDULUM_3 = (80, 80, 255)
    COLOR_ORB = (255, 215, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_ANCHOR = (200, 200, 220)
    COLOR_SELECTION = (255, 255, 0)
    COLOR_WIN = (255, 255, 0)
    COLOR_LOSE = (255, 0, 0)
    COLOR_PARTICLE_CATCH = (255, 255, 150)
    COLOR_PARTICLE_COLLIDE = (255, 100, 255)
    COLOR_PARTICLE_FAIL = (200, 0, 0)

    # Physics
    GRAVITY = 0.01
    DAMPING = 0.998
    PUSH_FORCE = 0.005
    INITIAL_ORB_SPEED = 2.0
    ORB_SPEED_INCREASE = 0.05

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.orbs_caught = 0
        self.selected_pendulum_idx = 0
        self.pendulums = []
        self.orbs = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.orbs_caught = 0
        self.selected_pendulum_idx = 0

        self._init_pendulums()
        self.orbs = []
        self._spawn_orb()
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if not self.game_over:
            if movement == 3: self.selected_pendulum_idx = 0  # Left
            elif movement == 2: self.selected_pendulum_idx = 1  # Down
            elif movement == 4: self.selected_pendulum_idx = 2  # Right

            self._update_pendulums(space_held, shift_held)
            self._update_orbs()

            event_reward = self._handle_collisions()
            reward += event_reward

            reward += 0.1 * len(self.orbs)

        self._update_particles()

        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.game_over:
            if self.orbs_caught >= self.ORBS_TO_WIN:
                reward += 100
            elif any(o['pos'][1] >= self.SCREEN_HEIGHT - o['radius'] for o in self.orbs):
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_pendulums()
        self._render_orbs()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orbs_caught": self.orbs_caught,
        }

    def _init_pendulums(self):
        self.pendulums = [
            {'angle': math.pi / 2, 'ang_vel': 0, 'length': 120, 'color': self.COLOR_PENDULUM_1, 'anchor': (160, 50), 'bob_radius': 15},
            {'angle': math.pi / 2, 'ang_vel': 0, 'length': 150, 'color': self.COLOR_PENDULUM_2, 'anchor': (320, 50), 'bob_radius': 15},
            {'angle': math.pi / 2, 'ang_vel': 0, 'length': 120, 'color': self.COLOR_PENDULUM_3, 'anchor': (480, 50), 'bob_radius': 15}
        ]

    def _spawn_orb(self):
        if self.orbs_caught + len(self.orbs) < self.ORBS_TO_WIN:
            x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            speed_multiplier = 1.0 + (self.orbs_caught // 2) * self.ORB_SPEED_INCREASE
            speed = self.INITIAL_ORB_SPEED * speed_multiplier
            self.orbs.append({
                'pos': np.array([x, 60.0]),
                'vel': np.array([0.0, speed]),
                'radius': 10,
                'color': self.COLOR_ORB
            })

    def _update_pendulums(self, push_left, push_right):
        for i, p in enumerate(self.pendulums):
            if i == self.selected_pendulum_idx:
                if push_left: p['ang_vel'] -= self.PUSH_FORCE
                if push_right: p['ang_vel'] += self.PUSH_FORCE

            ang_accel = -self.GRAVITY / p['length'] * math.sin(p['angle'] - math.pi / 2)
            p['ang_vel'] += ang_accel
            p['ang_vel'] *= self.DAMPING
            p['angle'] += p['ang_vel']
            p['angle'] = np.clip(p['angle'], math.pi / 4, 3 * math.pi / 4)

    def _update_orbs(self):
        for orb in self.orbs:
            orb['pos'] += orb['vel']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        orbs_to_remove = []
        bob_positions = [self._get_pendulum_bob_pos(p) for p in self.pendulums]

        for orb in self.orbs:
            for i, p in enumerate(self.pendulums):
                dist = np.linalg.norm(orb['pos'] - bob_positions[i])
                if dist < p['bob_radius'] + orb['radius']:
                    self.score += 1
                    self.orbs_caught += 1
                    reward += 10
                    orbs_to_remove.append(orb)
                    self._create_particles(bob_positions[i], self.COLOR_PARTICLE_CATCH, 30)
                    if self.orbs_caught >= self.ORBS_TO_WIN:
                        self.game_over = True
                    break
            if orb in orbs_to_remove: continue

            if orb['pos'][1] >= self.SCREEN_HEIGHT - orb['radius']:
                self.game_over = True
                orbs_to_remove.append(orb)
                self._create_particles(orb['pos'], self.COLOR_PARTICLE_FAIL, 50)

        for i in range(len(self.pendulums)):
            for j in range(i + 1, len(self.pendulums)):
                p1, p2 = self.pendulums[i], self.pendulums[j]
                dist = np.linalg.norm(bob_positions[i] - bob_positions[j])
                if dist < p1['bob_radius'] + p2['bob_radius']:
                    self.game_over = True
                    reward -= 5
                    mid_point = (bob_positions[i] + bob_positions[j]) / 2
                    self._create_particles(mid_point, self.COLOR_PARTICLE_COLLIDE, 50)

        if orbs_to_remove:
            self.orbs = [o for o in self.orbs if o not in orbs_to_remove]
            if not self.game_over:
                self._spawn_orb()
        
        return reward

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_pendulums(self):
        for i, p in enumerate(self.pendulums):
            bob_pos = self._get_pendulum_bob_pos(p)
            bob_pos_int = (int(bob_pos[0]), int(bob_pos[1]))
            anchor_pos_int = p['anchor']

            pygame.draw.aaline(self.screen, self.COLOR_ANCHOR, anchor_pos_int, bob_pos_int, 1)
            pygame.gfxdraw.filled_circle(self.screen, anchor_pos_int[0], anchor_pos_int[1], 5, self.COLOR_ANCHOR)
            
            glow_color = (*p['color'], 60)
            pygame.gfxdraw.filled_circle(self.screen, bob_pos_int[0], bob_pos_int[1], p['bob_radius'] + 4, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, bob_pos_int[0], bob_pos_int[1], p['bob_radius'], p['color'])
            pygame.gfxdraw.aacircle(self.screen, bob_pos_int[0], bob_pos_int[1], p['bob_radius'], p['color'])
            
            if i == self.selected_pendulum_idx and not self.game_over:
                pygame.gfxdraw.aacircle(self.screen, anchor_pos_int[0], anchor_pos_int[1], 10, self.COLOR_SELECTION)

    def _render_orbs(self):
        for orb in self.orbs:
            pos_int = (int(orb['pos'][0]), int(orb['pos'][1]))
            glow_color = (*orb['color'], 80)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], orb['radius'] + 4, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], orb['radius'], orb['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], orb['radius'], self.COLOR_UI_TEXT)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            size = int(p['radius'] * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p['color'], alpha), (size, size), size)
                self.screen.blit(s, (p['pos'][0] - size, p['pos'][1] - size))

    def _render_ui(self):
        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        orbs_surf = self.font_ui.render(f"Orbs: {self.orbs_caught} / {self.ORBS_TO_WIN}", True, self.COLOR_UI_TEXT)
        orbs_rect = orbs_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(orbs_surf, orbs_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            is_win = self.orbs_caught >= self.ORBS_TO_WIN
            msg = "YOU WIN!" if is_win else "GAME OVER"
            color = self.COLOR_WIN if is_win else self.COLOR_LOSE
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_pendulum_bob_pos(self, p):
        x = p['anchor'][0] + p['length'] * math.cos(p['angle'])
        y = p['anchor'][1] + p['length'] * math.sin(p['angle'])
        return np.array([x, y])

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life,
                'color': color, 'radius': self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # The environment is initialized with a dummy video driver for headless operation.
    # To play manually, we need to unset the dummy driver and re-initialize pygame.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit()
    pygame.init()

    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pendulum Orb Catcher")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_q:
                    running = False

        if terminated:
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            continue

        keys = pygame.key.get_pressed()
        movement = 0 
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()
    pygame.quit()