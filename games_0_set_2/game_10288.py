import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "A puzzle game where you rotate mirrors to reflect laser beams and destroy all targets."
    user_guide = "Use arrow keys to set the target rotation of the selected mirror. Use space to select the next mirror and shift to select the previous one."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000
    NUM_LEVELS = 12
    LASER_PULSE_INTERVAL = 50  # steps

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 55)
    COLOR_MIRROR = (0, 255, 150)
    COLOR_MIRROR_SELECTED = (255, 255, 0)
    COLOR_TARGET = (0, 180, 255)
    COLOR_LASER = (255, 50, 50)
    COLOR_LASER_GLOW = (255, 50, 50, 70)
    COLOR_PARTICLE = (255, 200, 50)
    COLOR_UI_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.total_targets_generated = 0
        self.total_targets_destroyed = 0
        self.emitters = []
        self.mirrors = []
        self.targets = []
        self.lasers = []
        self.particles = []
        self.selected_mirror_idx = 0
        self.laser_pulse_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1

        self.total_targets_generated = 0
        self.total_targets_destroyed = 0

        self.prev_space_held = False
        self.prev_shift_held = False

        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.level > self.NUM_LEVELS, self.steps >= self.MAX_STEPS, self._get_info()

        self.steps += 1

        step_reward = self._handle_input(action)
        step_reward += self._update_game_state()
        
        terminated = self.level > self.NUM_LEVELS
        truncated = self.steps >= self.MAX_STEPS
        
        final_reward = 0
        if terminated or truncated:
            self.game_over = True
            final_reward = self._calculate_final_reward()
            
        total_reward = step_reward + final_reward
        self.score += total_reward

        return (
            self._get_observation(),
            total_reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _setup_level(self):
        self.mirrors = []
        self.targets = []
        self.lasers = []
        self.particles = []

        self.emitters = [
            {'pos': pygame.math.Vector2(20, self.HEIGHT / 2), 'dir': pygame.math.Vector2(1, 0)},
            {'pos': pygame.math.Vector2(self.WIDTH - 20, self.HEIGHT / 2), 'dir': pygame.math.Vector2(-1, 0)},
        ]

        num_mirrors = 5 + self.level
        num_targets = 8 + int(8 * 0.05 * (self.level - 1))
        self.total_targets_generated += num_targets

        occupied_rects = []

        for _ in range(num_mirrors):
            mirror_placed = False
            while not mirror_placed:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(100, self.WIDTH - 100),
                    self.np_random.uniform(50, self.HEIGHT - 50),
                )
                size = 30
                rect = pygame.Rect(pos.x - size, pos.y - size, size * 2, size * 2)
                if not any(rect.colliderect(r) for r in occupied_rects):
                    self.mirrors.append({
                        'pos': pos,
                        'angle': self.np_random.uniform(0, 2 * math.pi),
                        'target_angle': self.np_random.uniform(0, 2 * math.pi),
                        'size': 25,
                    })
                    occupied_rects.append(rect)
                    mirror_placed = True

        for _ in range(num_targets):
            target_placed = False
            while not target_placed:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(50, self.WIDTH - 50),
                    self.np_random.uniform(50, self.HEIGHT - 50),
                )
                size = 20
                rect = pygame.Rect(pos.x - size, pos.y - size, size * 2, size * 2)
                if not any(rect.colliderect(r) for r in occupied_rects):
                    self.targets.append({
                        'pos': pos,
                        'radius': 10,
                        'hits_this_pulse': 0,
                    })
                    occupied_rects.append(rect)
                    target_placed = True

        self.selected_mirror_idx = 0
        self.laser_pulse_timer = self.LASER_PULSE_INTERVAL

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if space_held and not self.prev_space_held and len(self.mirrors) > 0:
            self.selected_mirror_idx = (self.selected_mirror_idx + 1) % len(self.mirrors)
        if shift_held and not self.prev_shift_held and len(self.mirrors) > 0:
            self.selected_mirror_idx = (self.selected_mirror_idx - 1 + len(self.mirrors)) % len(self.mirrors)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if len(self.mirrors) > 0:
            mirror = self.mirrors[self.selected_mirror_idx]
            if movement == 1:
                mirror['target_angle'] = math.pi / 2
            elif movement == 2:
                mirror['target_angle'] = math.pi
            elif movement == 3:
                mirror['target_angle'] = 3 * math.pi / 2
            elif movement == 4:
                mirror['target_angle'] = 0

        return 0

    def _update_game_state(self):
        frame_reward = 0

        for mirror in self.mirrors:
            diff = mirror['target_angle'] - mirror['angle']
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            mirror['angle'] += diff * 0.2

        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        self.laser_pulse_timer -= 1
        self.lasers.clear()
        for t in self.targets:
            t['hits_this_pulse'] = 0

        if self.laser_pulse_timer <= 0:
            self.laser_pulse_timer = self.LASER_PULSE_INTERVAL
            self._cast_all_rays()

            destroyed_this_pulse = []
            for i, target in reversed(list(enumerate(self.targets))):
                frame_reward += target['hits_this_pulse'] * 0.1
                if target['hits_this_pulse'] >= 3:
                    frame_reward += 1
                    destroyed_this_pulse.append(target)
                    self._create_explosion(target['pos'])
                    self.targets.pop(i)
                    self.total_targets_destroyed += 1

            if len(destroyed_this_pulse) >= 3:
                frame_reward += 5

        if not self.targets:
            frame_reward += 50
            self.level += 1
            if self.level <= self.NUM_LEVELS:
                self._setup_level()

        return frame_reward

    def _cast_all_rays(self):
        for emitter in self.emitters:
            ray_pos = pygame.math.Vector2(emitter['pos'])
            ray_dir = pygame.math.Vector2(emitter['dir'])

            for _ in range(15):
                intersections = []

                for mirror in self.mirrors:
                    p1 = mirror['pos'] + pygame.math.Vector2(math.cos(mirror['angle']), math.sin(mirror['angle'])) * mirror['size']
                    p2 = mirror['pos'] - pygame.math.Vector2(math.cos(mirror['angle']), math.sin(mirror['angle'])) * mirror['size']

                    den = (p1.x - p2.x) * (ray_pos.y - (ray_pos.y + ray_dir.y)) - (p1.y - p2.y) * (ray_pos.x - (ray_pos.x + ray_dir.x))
                    if den != 0:
                        t = ((p1.x - ray_pos.x) * (ray_pos.y - (ray_pos.y + ray_dir.y)) - (p1.y - ray_pos.y) * (ray_pos.x - (ray_pos.x + ray_dir.x))) / den
                        u = -((p1.x - p2.x) * (p1.y - ray_pos.y) - (p1.y - p2.y) * (p1.x - ray_pos.x)) / den
                        if 0 < t < 1 and u > 0:
                            intersect_pt = p1 + t * (p2 - p1)
                            dist = (intersect_pt - ray_pos).length()
                            if dist > 1e-5:
                                normal = pygame.math.Vector2(-math.sin(mirror['angle']), math.cos(mirror['angle']))
                                intersections.append({'type': 'mirror', 'dist': dist, 'point': intersect_pt, 'normal': normal})

                for target in self.targets:
                    oc = ray_pos - target['pos']
                    b = oc.dot(ray_dir)
                    c = oc.dot(oc) - target['radius'] ** 2
                    discriminant = b ** 2 - c
                    if discriminant >= 0:
                        dist = -b - math.sqrt(discriminant)
                        if dist > 1e-5:
                            intersections.append({'type': 'target', 'dist': dist, 'point': ray_pos + dist * ray_dir, 'obj': target})

                if not intersections:
                    end_point = ray_pos + ray_dir * 1000
                    self.lasers.append({'start': ray_pos, 'end': end_point})
                    break

                closest = min(intersections, key=lambda x: x['dist'])

                self.lasers.append({'start': ray_pos, 'end': closest['point']})

                if closest['type'] == 'mirror':
                    ray_pos = closest['point']
                    ray_dir = ray_dir.reflect(closest['normal'])
                elif closest['type'] == 'target':
                    closest['obj']['hits_this_pulse'] += 1
                    break

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(15, 31),
                'max_life': 30,
                'size': self.np_random.uniform(1, 4),
            })

    def _calculate_final_reward(self):
        if self.total_targets_generated > 0:
            accuracy = self.total_targets_destroyed / self.total_targets_generated
            if accuracy >= 0.9:
                return 100
            else:
                return -100
        return -100

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        for target in self.targets:
            pygame.gfxdraw.aacircle(self.screen, int(target['pos'].x), int(target['pos'].y), int(target['radius']), self.COLOR_TARGET)
            pygame.gfxdraw.filled_circle(self.screen, int(target['pos'].x), int(target['pos'].y), int(target['radius']), self.COLOR_TARGET)

        for i, mirror in enumerate(self.mirrors):
            color = self.COLOR_MIRROR_SELECTED if i == self.selected_mirror_idx else self.COLOR_MIRROR
            p1 = mirror['pos'] + pygame.math.Vector2(math.cos(mirror['angle']), math.sin(mirror['angle'])) * mirror['size']
            p2 = mirror['pos'] - pygame.math.Vector2(math.cos(mirror['angle']), math.sin(mirror['angle'])) * mirror['size']
            pygame.draw.line(self.screen, color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 3)
            if i == self.selected_mirror_idx:
                pygame.draw.line(self.screen, color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 7)

        for laser in self.lasers:
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, laser['start'], laser['end'], 5)
            pygame.draw.aaline(self.screen, self.COLOR_LASER, laser['start'], laser['end'], 2)

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*self.COLOR_PARTICLE, alpha)
            temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        level_text = self.font.render(f"Level: {self.level}/{self.NUM_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))

        accuracy = 0
        if self.total_targets_generated > 0:
            accuracy = (self.total_targets_destroyed / self.total_targets_generated) * 100
        acc_text = self.font.render(f"Total Acc: {accuracy:.1f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(acc_text, (self.WIDTH - acc_text.get_width() - 10, 10))

        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

    def _get_info(self):
        accuracy = 0
        if self.total_targets_generated > 0:
            accuracy = self.total_targets_destroyed / self.total_targets_generated
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "total_accuracy": accuracy,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()

    # The main loop needs a visible display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Laser Grid")
    clock = pygame.time.Clock()

    movement = 0
    space = 0
    shift = 0

    print("--- Manual Control ---")
    print("Arrow keys (Up/Down/Left/Right): Set mirror rotation to 90/180/270/0 degrees")
    print("Space: Select next mirror")
    print("Shift: Select previous mirror")
    print("Q: Quit")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                if event.key == pygame.K_SPACE:
                    space = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 0

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Accuracy: {info['total_accuracy']*100:.1f}%")
            obs, info = env.reset()
            movement, space, shift = 0, 0, 0

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()