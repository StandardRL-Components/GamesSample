import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


# Set Pygame to run in headless mode, which is required for the environment's backend.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A physics-based game where the player controls a 'quark' to collect other quarks.
    The player can move, shift gravity causing the entire level to rotate, and
    change their quark's size.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- Metadata for Gymnasium integration ---
    game_description = (
        "Collect quarks in a dynamic physics environment. Shift gravity and change "
        "your size to navigate obstacles and gather all the quarks."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to rotate gravity and shift to change your size."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = pygame.math.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    COLOR_BG = (20, 30, 50)
    COLOR_GRID = (30, 40, 60)
    COLOR_TERRAIN = (74, 74, 74)
    COLOR_PLAYER = (0, 255, 255)  # Cyan
    QUARK_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80)]  # Red, Green, Blue, Yellow
    COLOR_UI_TEXT = (255, 255, 255)

    PLAYER_SPEED = 4.0
    GRAVITY_FORCE = 0.5
    FRICTION = 0.95
    MAX_STEPS = 1000

    QUARK_SIZE_SMALL = 8
    QUARK_SIZE_LARGE = 16

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)

        # --- Game state variables ---
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_quark = {}
        self.quarks = []
        self.terrain_blocks = []
        self.particles = []

        self.gravity_idx = 0  # 0:Down, 1:Right, 2:Up, 3:Left
        self.gravity_vectors = [
            pygame.math.Vector2(0, 1),
            pygame.math.Vector2(1, 0),
            pygame.math.Vector2(0, -1),
            pygame.math.Vector2(-1, 0)
        ]

        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.game_over and len(self.quarks) == 0:  # Level was completed by collecting all quarks
            self.level += 1
        else:  # Reset from timeout or first start
            self.level = 1

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity_idx = 0

        self.particles.clear()

        self._generate_level()

        self.prev_space_held = True  # Prevent action on first frame
        self.prev_shift_held = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        reward = 0.0

        # --- Pre-action state for reward calculation ---
        dist_before = self._get_closest_quark_dist()

        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Physics ---
        self._update_physics()

        # --- Check for Collections ---
        collected_count = self._check_collections()
        if collected_count > 0:
            reward += 1.0 * collected_count
            self.score += collected_count

        # --- Post-action state for reward calculation ---
        dist_after = self._get_closest_quark_dist()
        if dist_before is not None and dist_after is not None:
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.1

        # --- Update Step Counter and Check for Termination ---
        self.steps += 1
        terminated = self._check_termination()
        truncated = False  # This environment does not truncate

        if terminated and not self.game_over:
            if len(self.quarks) == 0:  # Win condition
                reward += 10.0
                self.score += 10
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y -= self.PLAYER_SPEED  # Up
        elif movement == 2: move_vec.y += self.PLAYER_SPEED  # Down
        elif movement == 3: move_vec.x -= self.PLAYER_SPEED  # Left
        elif movement == 4: move_vec.x += self.PLAYER_SPEED  # Right
        self.player_quark['vel'] += move_vec

        if space_held and not self.prev_space_held:
            self._flip_gravity()

        if shift_held and not self.prev_shift_held:
            self.player_quark['is_large'] = not self.player_quark['is_large']
            new_size = self.QUARK_SIZE_LARGE if self.player_quark['is_large'] else self.QUARK_SIZE_SMALL
            self.player_quark['size'] = new_size
            self._create_particles(self.player_quark['pos'], self.COLOR_PLAYER, 20, is_ring=True)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_physics(self):
        all_quarks = [self.player_quark] + self.quarks
        gravity = self.gravity_vectors[self.gravity_idx] * self.GRAVITY_FORCE

        for quark in all_quarks:
            quark['vel'] += gravity
            quark['vel'] *= self.FRICTION

            quark_radius = quark['size']

            quark['pos'].x += quark['vel'].x
            for block in self.terrain_blocks:
                if block.clipline(quark['pos'].x - quark_radius, quark['pos'].y, quark['pos'].x + quark_radius, quark['pos'].y):
                    if quark['vel'].x > 0: quark['pos'].x = block.left - quark_radius
                    elif quark['vel'].x < 0: quark['pos'].x = block.right + quark_radius
                    quark['vel'].x = 0

            quark['pos'].y += quark['vel'].y
            for block in self.terrain_blocks:
                if block.clipline(quark['pos'].x, quark['pos'].y - quark_radius, quark['pos'].x, quark['pos'].y + quark_radius):
                    if quark['vel'].y > 0: quark['pos'].y = block.top - quark_radius
                    elif quark['vel'].y < 0: quark['pos'].y = block.bottom + quark_radius
                    quark['vel'].y = 0

            quark['pos'].x = max(quark_radius, min(self.SCREEN_WIDTH - quark_radius, quark['pos'].x))
            quark['pos'].y = max(quark_radius, min(self.SCREEN_HEIGHT - quark_radius, quark['pos'].y))

    def _check_collections(self):
        collected_count = 0
        player_pos = self.player_quark['pos']
        player_size = self.player_quark['size']

        remaining_quarks = []
        for q in self.quarks:
            dist = player_pos.distance_to(q['pos'])
            if dist < player_size + q['size']:
                self._create_particles(q['pos'], q['color'], 30)
                collected_count += 1
            else:
                remaining_quarks.append(q)

        self.quarks = remaining_quarks
        return collected_count

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or len(self.quarks) == 0

    def _flip_gravity(self):
        self.gravity_idx = (self.gravity_idx + 1) % 4

        all_entities = self.quarks + [self.player_quark]
        for entity in all_entities:
            relative_pos = entity['pos'] - self.CENTER
            entity['pos'] = pygame.math.Vector2(relative_pos.y, -relative_pos.x) + self.CENTER
            entity['vel'] = pygame.math.Vector2(entity['vel'].y, -entity['vel'].x)

        for block in self.terrain_blocks:
            relative_pos = pygame.math.Vector2(block.center) - self.CENTER
            new_center = pygame.math.Vector2(relative_pos.y, -relative_pos.x) + self.CENTER
            w, h = block.size
            block.size = (h, w)
            block.center = new_center

        self._create_particles(self.CENTER, (255, 255, 0), 100, speed_mult=2.0)

    def _generate_level(self):
        self.quarks.clear()
        self.terrain_blocks.clear()

        self.player_quark = {
            'pos': pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 4),
            'vel': pygame.math.Vector2(0, 0), 'size': self.QUARK_SIZE_SMALL,
            'is_large': False, 'color': self.COLOR_PLAYER
        }

        spawn_rect = pygame.Rect(50, 50, self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 100)

        num_quarks = 2 + self.level - 1
        for i in range(num_quarks):
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(spawn_rect.left, spawn_rect.right),
                    self.np_random.uniform(spawn_rect.top, spawn_rect.bottom)
                )
                if pos.distance_to(self.player_quark['pos']) > 50 and all(pos.distance_to(q['pos']) > 30 for q in self.quarks):
                    self.quarks.append({
                        'pos': pos, 'vel': pygame.math.Vector2(0, 0),
                        'size': self.QUARK_SIZE_SMALL,
                        'color': self.QUARK_COLORS[i % len(self.QUARK_COLORS)]
                    })
                    break

        num_blocks = 3 + (self.level - 1) * 2
        for _ in range(num_blocks):
            while True:
                w = self.np_random.integers(40, 150)
                h = self.np_random.integers(10, 30)
                if self.np_random.random() > 0.5: w, h = h, w

                block = pygame.Rect(
                    self.np_random.uniform(spawn_rect.left, spawn_rect.right - w),
                    self.np_random.uniform(spawn_rect.top, spawn_rect.bottom - h), w, h
                )

                if not block.collidepoint(self.player_quark['pos']) and \
                   all(not block.collidepoint(q['pos']) for q in self.quarks) and \
                   all(not block.colliderect(b) for b in self.terrain_blocks):
                    self.terrain_blocks.append(block)
                    break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_draw_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        for block in self.terrain_blocks: pygame.draw.rect(self.screen, self.COLOR_TERRAIN, block)

        for q in self.quarks:
            pos = (int(q['pos'].x), int(q['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], q['size'], q['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], q['size'], q['color'])

        p = self.player_quark
        pos = (int(p['pos'].x), int(p['pos'].y))
        glow_size = int(p['size'] * 1.5)
        glow_color = p['color'] + (50,)
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (pos[0] - glow_size, pos[1] - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p['size'], p['color'])
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['size'], p['color'])

    def _render_ui(self):
        score_text = f"SCORE: {self.score}  LEVEL: {self.level}"
        text_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        quarks_text = f"QUARKS: {len(self.quarks)}"
        text_surf = self.font.render(quarks_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        arrow_points = [pygame.math.Vector2(0, -15), pygame.math.Vector2(10, 0), pygame.math.Vector2(-10, 0)]
        center_pos = pygame.math.Vector2(30, self.SCREEN_HEIGHT - 30)
        angle = self.gravity_idx * -90
        rotated_points = [p.rotate(angle) + center_pos for p in arrow_points]

        pygame.gfxdraw.aapolygon(self.screen, rotated_points, (255, 255, 0))
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, (255, 255, 0))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "quarks_left": len(self.quarks)}

    def _get_closest_quark_dist(self):
        if not self.quarks: return None
        return min(self.player_quark['pos'].distance_to(q['pos']) for q in self.quarks)

    def _create_particles(self, pos, color, count, speed_mult=1.0, is_ring=False):
        for _ in range(count):
            if is_ring:
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed_mult * 2
            else:
                vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
                if vel.length() > 0:
                    vel = vel.normalize() * self.np_random.uniform(1, 3) * speed_mult
            # FIX: pygame.math.Vector2 does not have a .copy() method.
            # Create a new vector to copy it.
            self.particles.append({'pos': pygame.math.Vector2(pos), 'vel': vel, 'life': 1.0, 'color': color})

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 0.05
            if p['life'] > 0:
                remaining_particles.append(p)
                size = int(p['life'] * 4)
                if size > 0:
                    pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)
        self.particles = remaining_particles

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows for interactive play testing of the environment.
    # Note: Due to the headless setup (SDL_VIDEODRIVER="dummy"), no window will be displayed.
    # To play interactively, comment out the os.environ line at the top of the file.
    
    # Re-enable display for the main block if not running in a headless server
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quark Collector")
    clock = pygame.time.Clock()
    
    total_reward = 0
    done = False
    
    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(30)

    env.close()