import gymnasium as gym
import os
import pygame
import numpy as np
import math
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A musical puzzle platformer where the player, a cell, collects notes to form
    melodies. Playing melodies alters the environment (e.g., toggling platforms,
    reversing gravity) to help the cell navigate upwards to the goal.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a musical world as a cell, collecting notes to form melodies. "
        "Play melodies to alter the environment, toggle platforms, and even reverse gravity to reach the goal."
    )
    user_guide = (
        "Use ←→ arrow keys to move, ↑ to jump, and ↓ to fall faster. Press space to play the collected melody."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_PLAYER = (255, 220, 50)
    COLOR_PLAYER_GLOW = (255, 220, 50, 50)
    COLOR_PLATFORM_1 = (50, 200, 150)
    COLOR_PLATFORM_2 = (200, 50, 150)
    COLOR_PLATFORM_STATIC = (100, 120, 140)
    COLOR_PLATFORM_INACTIVE = (60, 70, 80)
    COLOR_GOAL = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    NOTE_COLORS = [(0, 150, 255), (255, 100, 0), (220, 0, 220), (50, 255, 50)]

    # Physics
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = -0.15
    PLAYER_JUMP_STRENGTH = -12
    GRAVITY_STRENGTH = 0.6
    MAX_VEL_X = 6
    MAX_VEL_Y = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State Initialization ---
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_radius = 12
        self.on_ground = False
        self.last_space_held = False

        self.platforms = []
        self.notes = []
        self.goal_rect = pygame.Rect(0, 0, 0, 0)

        self.collected_notes = []
        self.max_notes = 4
        self.melodies = {
            (0, 1): "TOGGLE_1",
            (2, 0, 3): "TOGGLE_2",
            (1, 1, 1): "REVERSE_GRAVITY",
        }

        self.gravity = pygame.Vector2(0, self.GRAVITY_STRENGTH)
        self.particles = []
        self.background_circles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Player State ---
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.last_space_held = False

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collected_notes = []
        self.gravity = pygame.Vector2(0, self.GRAVITY_STRENGTH)
        self.particles = []
        self._generate_level()

        # Generate static background
        self.background_circles = []
        for _ in range(50):
            self.background_circles.append((
                self.np_random.integers(0, self.WIDTH + 1),
                self.np_random.integers(0, self.HEIGHT + 1),
                self.np_random.integers(5, 51),
                (self.np_random.integers(20, 41), self.np_random.integers(30, 51), self.np_random.integers(50, 71))
            ))

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.notes = []

        # Starting platform
        self.platforms.append({'rect': pygame.Rect(200, self.HEIGHT - 30, 240, 20), 'type': 'static', 'active': True})

        # Level layout (x, y, w, h, type, active)
        platform_data = [
            (100, 300, 100, 15, 'static', True),
            (440, 300, 100, 15, 'static', True),
            (270, 220, 100, 15, 1, True),
            (50, 150, 80, 15, 'static', True),
            (510, 150, 80, 15, 'static', True),
            (180, 80, 120, 15, 2, True),
            (340, 80, 120, 15, 2, False),
            (270, 0, 100, 15, 'static', True), # Part of goal
        ]
        for p in platform_data:
            self.platforms.append({'rect': pygame.Rect(p[0], p[1], p[2], p[3]), 'type': p[4], 'active': p[5]})

        # Note layout (x, y, type_id)
        note_data = [
            (150, 270, 0), # A
            (490, 270, 1), # B
            (70, 120, 2), # C
            (530, 120, 0), # A
            (240, 50, 3), # D
            (150, 350, 1), # B (easy access for gravity)
            (490, 350, 1), # B (easy access for gravity)
        ]
        for n in note_data:
            self.notes.append({'pos': pygame.Vector2(n[0], n[1]), 'type': n[2], 'radius': 8})

        # Goal
        self.goal_rect = pygame.Rect(270, -10, 100, 25)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        old_y = self.player_pos.y

        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held)

        # --- Update Game Logic ---
        self._update_player_physics()
        self._handle_collisions()
        self._update_particles()

        # --- Calculate Rewards ---
        # Small penalty for existing
        self.reward_this_step -= 0.01
        # Reward for upward progress
        height_gain = old_y - self.player_pos.y
        self.reward_this_step += height_gain * 0.02

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.player_pos.y > self.HEIGHT + self.player_radius or self.player_pos.y < -self.player_radius - 50:
            # Fell off top or bottom
            self.reward_this_step -= 100
            self.game_over = True
            terminated = True
        elif self.goal_rect.colliderect(self._get_player_rect()):
            self.reward_this_step += 100
            self.score += 100 # Add to score as well
            self.game_over = True
            terminated = True

        self.steps += 1
        if self.steps >= 1500: # Increased step limit for more exploration time
            truncated = True
            terminated = True

        self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL

        # Vertical movement (Jump/Fast-Fall)
        if movement == 1 and self.on_ground:  # Up (Jump)
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH * (1 if self.gravity.y > 0 else -1)
        elif movement == 2:  # Down (Fast-Fall)
            self.player_vel.y += self.gravity.y * 1.5

        # Play melody on space press (rising edge)
        if space_held and not self.last_space_held:
            self._play_melody()
        self.last_space_held = space_held

    def _play_melody(self):
        if not self.collected_notes:
            return

        melody_tuple = tuple(self.collected_notes)
        effect = self.melodies.get(melody_tuple)

        if effect:
            self.reward_this_step += 1.0

            if effect == "TOGGLE_1":
                for p in self.platforms:
                    if p['type'] == 1:
                        p['active'] = not p['active']
            elif effect == "TOGGLE_2":
                for p in self.platforms:
                    if p['type'] == 2:
                        p['active'] = not p['active']
            elif effect == "REVERSE_GRAVITY":
                self.gravity.y *= -1

            # Spawn particles for feedback
            for _ in range(30):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                self.particles.append({
                    'pos': self.player_pos.copy(),
                    'vel': vel,
                    'lifetime': self.np_random.integers(15, 31),
                    'radius': self.np_random.uniform(2, 5),
                    'color': self.NOTE_COLORS[self.np_random.integers(0, len(self.NOTE_COLORS))]
                })
        else:
            self.reward_this_step -= 0.1

        self.collected_notes = []


    def _update_player_physics(self):
        # Apply friction
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0

        # Apply gravity
        self.player_vel += self.gravity
        self.on_ground = False

        # Clamp velocities
        self.player_vel.x = max(-self.MAX_VEL_X, min(self.MAX_VEL_X, self.player_vel.x))
        self.player_vel.y = max(-self.MAX_VEL_Y, min(self.MAX_VEL_Y, self.player_vel.y))

        # Update position
        self.player_pos += self.player_vel

        # Screen bounds (horizontal)
        if self.player_pos.x - self.player_radius < 0:
            self.player_pos.x = self.player_radius
            self.player_vel.x = 0
        if self.player_pos.x + self.player_radius > self.WIDTH:
            self.player_pos.x = self.WIDTH - self.player_radius
            self.player_vel.x = 0

    def _get_player_rect(self):
        return pygame.Rect(
            self.player_pos.x - self.player_radius,
            self.player_pos.y - self.player_radius,
            self.player_radius * 2,
            self.player_radius * 2
        )

    def _handle_collisions(self):
        player_rect = self._get_player_rect()

        # --- Platform Collisions ---
        for p in self.platforms:
            if p['active'] and player_rect.colliderect(p['rect']):
                # Vertical collision check
                # Player is moving down and was above the platform in the previous frame
                if self.player_vel.y > 0 and (player_rect.bottom - self.player_vel.y) <= p['rect'].top:
                    self.player_pos.y = p['rect'].top - self.player_radius
                    self.player_vel.y = 0
                    self.on_ground = True
                # Player is moving up and was below the platform in the previous frame
                elif self.player_vel.y < 0 and (player_rect.top - self.player_vel.y) >= p['rect'].bottom:
                    self.player_pos.y = p['rect'].bottom + self.player_radius
                    self.player_vel.y = 0
                # Horizontal collision
                else:
                    if self.player_vel.x > 0 and player_rect.right - self.player_vel.x <= p['rect'].left:
                         self.player_pos.x = p['rect'].left - self.player_radius
                         self.player_vel.x = 0
                    elif self.player_vel.x < 0 and player_rect.left - self.player_vel.x >= p['rect'].right:
                         self.player_pos.x = p['rect'].right + self.player_radius
                         self.player_vel.x = 0


        # --- Note Collection ---
        for note in self.notes[:]:
            if self.player_pos.distance_to(note['pos']) < self.player_radius + note['radius']:
                if len(self.collected_notes) < self.max_notes:
                    self.collected_notes.append(note['type'])
                    self.notes.remove(note)
                    self.reward_this_step += 0.5 # Increased reward for collecting a note
                    # SFX: Note Collect

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] *= 0.98
            if p['lifetime'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x, y, r, color in self.background_circles:
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(r), color)

    def _render_game(self):
        # Draw Goal
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_rect)
        pygame.gfxdraw.rectangle(self.screen, self.goal_rect, (*self.COLOR_GOAL, 150))

        # Draw Platforms
        for p in self.platforms:
            color = self.COLOR_PLATFORM_INACTIVE
            if p['active']:
                if p['type'] == 'static': color = self.COLOR_PLATFORM_STATIC
                elif p['type'] == 1: color = self.COLOR_PLATFORM_1
                elif p['type'] == 2: color = self.COLOR_PLATFORM_2

            pygame.draw.rect(self.screen, color, p['rect'], border_radius=3)
            if p['active']:
                pygame.draw.rect(self.screen, tuple(min(255, c + 30) for c in color[:3]), p['rect'].inflate(-6, -6), border_radius=3)


        # Draw Notes
        for n in self.notes:
            pos = (int(n['pos'].x), int(n['pos'].y))
            color = self.NOTE_COLORS[n['type']]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(n['radius']), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(n['radius']), color)

        # Draw Particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            alpha = int(255 * (p['lifetime'] / 30))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Draw Player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], int(self.player_radius * 1.5), self.COLOR_PLAYER_GLOW)
        # Body
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{1500}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Collected Notes
        for i, note_type in enumerate(self.collected_notes):
            x = self.WIDTH // 2 - (len(self.collected_notes) * 30) // 2 + i * 30
            y = self.HEIGHT - 25
            color = self.NOTE_COLORS[note_type]
            pygame.draw.rect(self.screen, color, (x, y, 20, 20), border_radius=4)
            note_char = ['A', 'B', 'C', 'D'][note_type]
            note_text = self.font_small.render(note_char, True, (255,255,255))
            self.screen.blit(note_text, (x + 6, y + 2))

        # Gravity Indicator
        arrow_start = pygame.Vector2(20, self.HEIGHT / 2)
        if self.gravity.y != 0:
            arrow_end = arrow_start + self.gravity.normalize() * 20
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, arrow_start, arrow_end, 2)
            p1 = arrow_end + self.gravity.normalize().rotate(150) * 8
            p2 = arrow_end + self.gravity.normalize().rotate(-150) * 8
            pygame.draw.polygon(self.screen, self.COLOR_UI_TEXT, [arrow_end, p1, p2])


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "collected_notes": len(self.collected_notes),
            "gravity_y": self.gravity.y
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play ---
    env = GameEnv()
    obs, info = env.reset(seed=42)

    # For human playback, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Musical Cell Platformer")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # This action is unused in the current logic

        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset(seed=42)
                total_reward = 0
                terminated = False
                truncated = False

        if terminated:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_DOWN]:
            movement = 2
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Info: {info}")
    env.close()