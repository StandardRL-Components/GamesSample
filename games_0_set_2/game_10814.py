import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete, Box


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a hazardous vertical shaft by flipping gravity and building temporary platforms. "
        "Dodge falling debris and reach the exit portal before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ to move and ↑ to jump. Press space to flip gravity and shift to build a platform."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_CONDUIT = (50, 200, 255)
    COLOR_CONDUIT_GLOW = (50, 150, 200)
    COLOR_DEBRIS = (255, 80, 50)
    COLOR_DEBRIS_GLOW = (200, 60, 40)
    COLOR_EXIT = (100, 255, 100)
    COLOR_EXIT_GLOW = (80, 200, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PARTICLE_EXPLOSION = [(255, 100, 50), (255, 150, 50), (200, 80, 40)]
    COLOR_PARTICLE_BUILD = [(50, 200, 255), (100, 220, 255)]
    COLOR_PARTICLE_FLIP = [(200, 100, 255), (150, 80, 220)]
    COLOR_BG_GRID = (25, 30, 45)

    # Physics & Gameplay
    GRAVITY_ACCEL = 0.4
    PLAYER_ACCEL = 0.8
    PLAYER_JUMP_FORCE = 10.0
    PLAYER_FRICTION = -0.12
    MAX_GRAVITY_FLIPS = 8
    CONDUIT_SIZE = (60, 10)
    CONDUIT_LIFESPAN = 450  # 15 seconds at 30fps
    CONDUIT_COOLDOWN_STEPS = 15  # 0.5 seconds
    DEBRIS_SPAWN_BASE_RATE = 0.01
    DEBRIS_SPAWN_INCREASE = 0.005  # Rate increases by this much per 100 steps
    DEBRIS_SIZE_RANGE = (15, 30)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        self.render_mode = render_mode

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.gravity_direction = None
        self.gravity_flips_remaining = None
        self.on_ground = None

        self.conduits = None
        self.debris = None
        self.particles = None

        self.exit_portal_rect = None

        self.steps = None
        self.score = None
        self.game_over = None

        self.last_space_held = None
        self.last_shift_held = None
        self.last_movement_dir = None
        self.conduit_cooldown = None
        self.max_progress = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.gravity_direction = 1  # 1 for down, -1 for up
        self.gravity_flips_remaining = self.MAX_GRAVITY_FLIPS

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 20, 20)
        self.player_rect.center = self.player_pos
        self.on_ground = False

        self.exit_portal_rect = pygame.Rect(self.SCREEN_WIDTH / 2 - 25, 20, 50, 20)

        self.conduits = []
        # Add a permanent ground floor
        ground = {'rect': pygame.Rect(0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10), 'life': float('inf')}
        self.conduits.append(ground)

        self.debris = []
        self.particles = []

        self.last_space_held = False
        self.last_shift_held = False
        self.last_movement_dir = pygame.Vector2(0, 1)  # Default to down
        self.conduit_cooldown = 0

        self.max_progress = self._get_player_progress()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False

        old_dist_to_exit = self._get_dist_to_exit()

        self._handle_input(action)
        self._update_player()
        self._update_debris()
        self._update_conduits_and_particles()

        new_dist_to_exit = self._get_dist_to_exit()
        reward += (old_dist_to_exit - new_dist_to_exit) * 0.1  # Reward for getting closer

        # Progress reward
        current_progress = self._get_player_progress()
        if current_progress < self.max_progress:
            reward += 5.0
            self.max_progress = current_progress

        self.score += reward
        self.steps += 1

        # Check termination conditions
        if self.player_rect.colliderect(self.exit_portal_rect):
            reward += 100
            terminated = True
            self.game_over = True
        elif not self.screen.get_rect().colliderect(self.player_rect):
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Truncated in a wrapper
            self.game_over = True

        # Debris collision check happens in _update_debris, sets self.game_over
        if self.game_over and not terminated:  # Crushed by debris
            reward -= 100
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is handled by wrappers
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Player Movement
        if movement == 1:  # Up
            if self.on_ground:
                self.player_vel.y = -self.PLAYER_JUMP_FORCE * self.gravity_direction
            self.last_movement_dir = pygame.Vector2(0, -1)
        elif movement == 2:  # Down
            self.last_movement_dir = pygame.Vector2(0, 1)
        elif movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
            self.last_movement_dir = pygame.Vector2(-1, 0)
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL
            self.last_movement_dir = pygame.Vector2(1, 0)

        # Gravity Flip
        if space_held and not self.last_space_held and self.gravity_flips_remaining > 0:
            self.gravity_direction *= -1
            self.gravity_flips_remaining -= 1
            self.score -= 1  # Penalty for using flip
            self._create_particles(self.player_pos, 30, self.COLOR_PARTICLE_FLIP, 2, 5, 60)

        # Conduit Placement
        if self.conduit_cooldown > 0:
            self.conduit_cooldown -= 1

        if shift_held and not self.last_shift_held and self.conduit_cooldown == 0:
            self.conduit_cooldown = self.CONDUIT_COOLDOWN_STEPS
            build_dir = self.last_movement_dir
            if movement in [1, 2, 3, 4]:  # Prioritize current direction input
                if movement == 1: build_dir = pygame.Vector2(0, -1)
                elif movement == 2: build_dir = pygame.Vector2(0, 1)
                elif movement == 3: build_dir = pygame.Vector2(-1, 0)
                elif movement == 4: build_dir = pygame.Vector2(1, 0)

            conduit_pos = self.player_pos + build_dir * 40

            # Use horizontal or vertical conduit based on direction
            if abs(build_dir.x) > abs(build_dir.y):  # Horizontal
                conduit_rect = pygame.Rect(0, 0, self.CONDUIT_SIZE[0], self.CONDUIT_SIZE[1])
            else:  # Vertical
                conduit_rect = pygame.Rect(0, 0, self.CONDUIT_SIZE[1], self.CONDUIT_SIZE[0])

            conduit_rect.center = conduit_pos

            # Prevent placing outside screen or inside other conduits
            if self.screen.get_rect().contains(conduit_rect) and conduit_rect.collidelist(
                    [c['rect'] for c in self.conduits]) == -1:
                self.conduits.append({'rect': conduit_rect, 'life': self.CONDUIT_LIFESPAN})
                self._create_particles(conduit_rect.center, 20, self.COLOR_PARTICLE_BUILD, 1, 3, 40)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY_ACCEL * self.gravity_direction

        # Apply friction
        self.player_vel.x *= (1.0 + self.PLAYER_FRICTION)
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

        # Limit speed
        self.player_vel.x = max(-10, min(10, self.player_vel.x))
        self.player_vel.y = max(-15, min(15, self.player_vel.y))

        # Move and handle collisions
        self.on_ground = False
        self.player_pos.x += self.player_vel.x
        self.player_rect.centerx = int(self.player_pos.x)
        self._handle_collisions('x')

        self.player_pos.y += self.player_vel.y
        self.player_rect.centery = int(self.player_pos.y)
        self._handle_collisions('y')

    def _handle_collisions(self, axis):
        colliding_conduits = [c['rect'] for c in self.conduits]

        for rect in colliding_conduits:
            if self.player_rect.colliderect(rect):
                if axis == 'x':
                    if self.player_vel.x > 0: self.player_rect.right = rect.left
                    elif self.player_vel.x < 0: self.player_rect.left = rect.right
                    self.player_pos.x = self.player_rect.centerx
                    self.player_vel.x = 0
                elif axis == 'y':
                    if self.player_vel.y * self.gravity_direction > 0:  # Moving towards surface
                        if self.gravity_direction == 1:  # Gravity down
                            self.player_rect.bottom = rect.top
                        else:  # Gravity up
                            self.player_rect.top = rect.bottom
                        self.player_vel.y = 0
                        self.on_ground = True
                    elif self.player_vel.y * self.gravity_direction < 0:  # Hitting ceiling
                        if self.gravity_direction == 1:  # Gravity down
                            self.player_rect.top = rect.bottom
                        else:  # Gravity up
                            self.player_rect.bottom = rect.top
                        self.player_vel.y = 0
                    self.player_pos.y = self.player_rect.centery

    def _update_debris(self):
        spawn_rate = self.DEBRIS_SPAWN_BASE_RATE + (self.steps // 100) * self.DEBRIS_SPAWN_INCREASE
        if self.np_random.random() < spawn_rate:
            size = self.np_random.integers(self.DEBRIS_SIZE_RANGE[0], self.DEBRIS_SIZE_RANGE[1])
            pos_x = self.np_random.random() * self.SCREEN_WIDTH
            pos_y = 0 if self.gravity_direction == 1 else self.SCREEN_HEIGHT
            debris_rect = pygame.Rect(pos_x, pos_y, size, size)
            self.debris.append({'rect': debris_rect, 'vel_y': 0})

        for d in self.debris[:]:
            d['vel_y'] += self.GRAVITY_ACCEL * self.gravity_direction * 0.8  # Debris falls slightly slower
            d['rect'].y += int(d['vel_y'])

            # Player collision
            if d['rect'].colliderect(self.player_rect):
                self.game_over = True
                self._create_particles(self.player_rect.center, 50, self.COLOR_PARTICLE_EXPLOSION, 3, 6, 80)
                self.debris.remove(d)
                continue

            # Conduit collision
            collided_conduits = [c for c in self.conduits if
                                 c['rect'].colliderect(d['rect']) and c['life'] != float('inf')]
            if collided_conduits:
                for c in collided_conduits:
                    self.conduits.remove(c)
                self._create_particles(d['rect'].center, 30, self.COLOR_PARTICLE_EXPLOSION, 2, 4, 60)
                self.debris.remove(d)
                continue

            # Off-screen removal
            if d['rect'].top > self.SCREEN_HEIGHT or d['rect'].bottom < 0:
                self.debris.remove(d)

    def _update_conduits_and_particles(self):
        for c in self.conduits[:]:
            if c['life'] != float('inf'):
                c['life'] -= 1
                if c['life'] <= 0:
                    self.conduits.remove(c)

        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.98
            if p['life'] <= 0 or p['radius'] < 1:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_BG_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_game(self):
        # Exit Portal
        self._render_glow_rect(self.exit_portal_rect, self.COLOR_EXIT, self.COLOR_EXIT_GLOW, 15)

        # Debris
        for d in self.debris:
            self._render_glow_rect(d['rect'], self.COLOR_DEBRIS, self.COLOR_DEBRIS_GLOW, 10)

        # Conduits
        for c in self.conduits:
            life_alpha = 255 if c['life'] == float('inf') else max(0, min(255, int(c['life'] / self.CONDUIT_LIFESPAN * 255)))
            color = list(self.COLOR_CONDUIT) + [life_alpha]
            glow_color = list(self.COLOR_CONDUIT_GLOW) + [life_alpha]
            self._render_glow_rect(c['rect'], color, glow_color, 10)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] / p['max_life'] * 255)))
            color = p['color']
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color + (alpha,))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color + (alpha,))

        # Player
        if not self.game_over or self.player_rect.colliderect(self.exit_portal_rect):
            self._render_glow_rect(self.player_rect, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 20)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        flips_text = self.font_ui.render(f"Gravity Flips: {self.gravity_flips_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(flips_text, (self.SCREEN_WIDTH - flips_text.get_width() - 10, 10))

        if self.game_over:
            status_text_str = "VICTORY!" if self.player_rect.colliderect(self.exit_portal_rect) else "GAME OVER"
            status_text = self.font_game_over.render(status_text_str, True, self.COLOR_UI_TEXT)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(status_text, text_rect)

    def _render_glow_rect(self, rect, color, glow_color, glow_size):
        glow_rect = rect.inflate(glow_size, glow_size)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)

        # Determine if color has alpha
        if len(color) == 4:
            base_color = tuple(color[:3])
            alpha = color[3]
            current_glow_color = tuple(glow_color[:3]) + (alpha,)
            pygame.draw.rect(glow_surf, current_glow_color, glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(self.screen, base_color, rect, border_radius=3)
        else:
            pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gravity_flips_remaining": self.gravity_flips_remaining
        }

    def _get_dist_to_exit(self):
        return self.player_pos.distance_to(pygame.Vector2(self.exit_portal_rect.center))

    def _get_player_progress(self):
        # Progress is the vertical distance to the exit, accounting for gravity
        if self.gravity_direction == 1:  # Normal gravity, exit is up
            return self.player_rect.centery
        else:  # Flipped gravity, exit is down
            return self.SCREEN_HEIGHT - self.player_rect.centery

    def _create_particles(self, pos, count, colors, min_speed, max_speed, max_life):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = self.np_random.uniform(3, 7)
            color = random.choice(colors)
            life = self.np_random.integers(max_life // 2, max_life)
            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': vel, 'radius': radius,
                'color': color, 'life': life, 'max_life': life
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium environment
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Create a display for human interaction
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Conduit")
    clock = pygame.time.Clock()

    while not done:
        movement, space, shift = 0, 0, 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Key state handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        elif keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)  # Run at 30 FPS for human play

    env.close()