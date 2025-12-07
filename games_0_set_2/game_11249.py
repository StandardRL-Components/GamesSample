import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame
import pygame.gfxdraw
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a neon data stream, timing your energy surges to absorb incoming data pulses while deflecting malicious viruses."
    )
    user_guide = (
        "Use the ← and → arrow keys to move. Press space to activate an energy surge to absorb data pulses and destroy viruses."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 5000
        self.PLAYER_Y_POS = self.SCREEN_HEIGHT - 50
        self.PLAYER_SPEED = 12
        self.PLAYER_HITBOX_RADIUS = 15
        self.SURGE_DURATION = 10  # frames
        self.SURGE_COOLDOWN = 15  # frames
        self.SURGE_MAX_RADIUS = 60
        self.STREAM_WIDTH = 10
        self.MAX_STREAMS = 7
        self.PULSE_HEIGHT = 20
        self.PULSE_HIT_WINDOW = 25  # pixels

        # Colors (Cyberpunk Neon)
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_PLAYER = (0, 255, 255)  # Cyan
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_STREAM_BLUE = (0, 100, 255)
        self.COLOR_STREAM_GREEN = (50, 255, 150)
        self.COLOR_PULSE = (255, 255, 255)
        self.COLOR_SURGE = (255, 255, 0)  # Yellow
        self.COLOR_VIRUS = (255, 20, 20)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_COMBO_TEXT = (255, 200, 0)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_combo = pygame.font.Font(None, 48)
        self.font_game_over = pygame.font.Font(None, 72)

        # Initialize state variables to be defined in reset()
        self.steps = None
        self.score = None
        self.health = None
        self.combo = None
        self.skill_points = None
        self.game_over = None
        self.player_x = None
        self.streams = None
        self.viruses = None
        self.particles = None
        self.stream_speed = None
        self.virus_spawn_prob = None
        self.surge_state = None
        self.space_was_held = None
        self.damage_flash_timer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.health = 3
        self.combo = 1
        self.skill_points = 0
        self.game_over = False

        self.player_x = self.SCREEN_WIDTH // 2

        self.streams = []
        self.viruses = []
        self.particles = []

        self.stream_speed = 2.5
        self.virus_spawn_prob = 0.01

        self.surge_state = {'active': False, 'timer': 0, 'cooldown': 0}
        self.space_was_held = False
        self.damage_flash_timer = 0

        # Initial stream generation
        for _ in range(self.MAX_STREAMS):
            self._spawn_stream()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Game State
        self._update_surge()
        reward += self._update_streams()
        reward += self._update_viruses()
        self._update_particles()
        self._update_difficulty()

        # 3. Spawn new entities
        if self.np_random.random() < self.virus_spawn_prob:
            self._spawn_virus()

        if len(self.streams) < self.MAX_STREAMS:
            self._spawn_stream(at_top=True)

        # 4. Check for termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            if truncated and not terminated: # Survival bonus only if not dead
                reward += 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 3:  # Left
            self.player_x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_x += self.PLAYER_SPEED

        self.player_x = np.clip(self.player_x, self.PLAYER_HITBOX_RADIUS,
                                self.SCREEN_WIDTH - self.PLAYER_HITBOX_RADIUS)

        # Activate surge on the rising edge of the space press
        if space_pressed and not self.space_was_held and self.surge_state['cooldown'] == 0:
            self.surge_state['active'] = True
            self.surge_state['timer'] = self.SURGE_DURATION
            self.surge_state['cooldown'] = self.SURGE_COOLDOWN + self.SURGE_DURATION
            # sfx: Power Surge Activate

        self.space_was_held = space_pressed

    def _update_surge(self):
        if self.surge_state['timer'] > 0:
            self.surge_state['timer'] -= 1
        else:
            self.surge_state['active'] = False

        if self.surge_state['cooldown'] > 0:
            self.surge_state['cooldown'] -= 1

    def _update_streams(self):
        step_reward = 0
        for stream in self.streams:
            pulses_to_remove = []
            for pulse in stream['pulses']:
                pulse['y'] += self.stream_speed

                # Check for pulse interaction
                if abs(pulse['y'] - self.PLAYER_Y_POS) < self.PULSE_HIT_WINDOW and pulse['active']:
                    is_aligned = abs(stream['x'] - self.player_x) < self.STREAM_WIDTH / 2 + self.PLAYER_HITBOX_RADIUS
                    if is_aligned and self.surge_state['active']:
                        # Successful match
                        pulse['active'] = False
                        step_reward += 1
                        self.score += 10 * self.combo
                        self.combo += 1
                        self._create_particles(stream['x'], self.PLAYER_Y_POS, self.COLOR_PULSE, 20)
                        # sfx: Pulse Match Success
                        if self.score // 1000 > self.skill_points:
                            self.skill_points = self.score // 1000
                            # sfx: Skill Point Earned

                if pulse['y'] > self.SCREEN_HEIGHT:
                    pulses_to_remove.append(pulse)

            stream['pulses'] = [p for p in stream['pulses'] if p not in pulses_to_remove]

            # Generate new pulses
            stream['pulse_timer'] -= 1
            if stream['pulse_timer'] <= 0:
                stream['pulses'].append({'y': 0, 'active': True})
                stream['pulse_timer'] = stream['pulse_interval']
        return step_reward

    def _update_viruses(self):
        step_reward = 0
        viruses_to_remove = []
        for virus in self.viruses:
            virus['y'] += self.stream_speed * 1.2  # Viruses are slightly faster

            dist_to_player = math.hypot(virus['x'] - self.player_x, virus['y'] - self.PLAYER_Y_POS)

            if self.surge_state['active']:
                surge_radius = self.SURGE_MAX_RADIUS * (1 - self.surge_state['timer'] / self.SURGE_DURATION)
                if dist_to_player < surge_radius:
                    step_reward += 5
                    self._create_particles(virus['x'], virus['y'], self.COLOR_VIRUS, 30)
                    viruses_to_remove.append(virus)
                    # sfx: Virus Deflected
                    continue

            if dist_to_player < self.PLAYER_HITBOX_RADIUS:
                self.health -= 1
                self.combo = 1  # Reset combo
                self.damage_flash_timer = 5
                viruses_to_remove.append(virus)
                # sfx: Player Hit
                continue

            if virus['y'] > self.SCREEN_HEIGHT:
                viruses_to_remove.append(virus)

        self.viruses = [v for v in self.viruses if v not in viruses_to_remove]
        return step_reward

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
            if p['lifespan'] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _update_difficulty(self):
        # Data stream speed increases by 0.05 every 500 steps.
        if self.steps > 0 and self.steps % 500 == 0:
            self.stream_speed = min(8, self.stream_speed + 0.05)

        # Virus attack frequency increases by 0.02 every 250 steps.
        if self.steps > 0 and self.steps % 250 == 0:
            self.virus_spawn_prob = min(0.1, self.virus_spawn_prob + 0.02)

    def _check_termination(self):
        return self.health <= 0

    def _spawn_stream(self, at_top=False):
        x_pos = self.np_random.integers(self.STREAM_WIDTH, self.SCREEN_WIDTH - self.STREAM_WIDTH)
        color = self.COLOR_STREAM_BLUE if self.np_random.random() > 0.5 else self.COLOR_STREAM_GREEN
        interval = self.np_random.integers(60, 120)

        initial_pulses = []
        if not at_top:
            # Pre-populate streams on reset
            num_pulses = self.np_random.integers(1, 4)
            for i in range(num_pulses):
                initial_pulses.append({'y': self.np_random.random() * self.SCREEN_HEIGHT, 'active': True})

        self.streams.append({
            'x': x_pos,
            'color': color,
            'pulses': initial_pulses,
            'pulse_interval': interval,
            'pulse_timer': self.np_random.integers(0, interval)
        })

    def _spawn_virus(self):
        x_pos = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
        self.viruses.append({'x': x_pos, 'y': 0})

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.random() * 3 + 2
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background_grid()
        self._render_streams()
        self._render_viruses()
        self._render_particles()
        self._render_player()
        self._render_surge()

        if self.damage_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 0, 0, 80))
            self.screen.blit(flash_surface, (0, 0))
            self.damage_flash_timer -= 1

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_streams(self):
        for stream in self.streams:
            # Faded stream line
            pygame.draw.line(self.screen, stream['color'], (stream['x'], 0), (stream['x'], self.SCREEN_HEIGHT), 1)
            for pulse in stream['pulses']:
                if pulse['active']:
                    y = int(pulse['y'])
                    x = int(stream['x'])
                    rect = pygame.Rect(x - self.STREAM_WIDTH // 2, y - self.PULSE_HEIGHT // 2, self.STREAM_WIDTH,
                                      self.PULSE_HEIGHT)

                    # Glow effect
                    glow_rect = rect.inflate(10, 10)
                    glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, (*self.COLOR_PULSE, 30), glow_surf.get_rect(), border_radius=5)
                    self.screen.blit(glow_surf, glow_rect.topleft)

                    pygame.draw.rect(self.screen, self.COLOR_PULSE, rect, border_radius=3)

    def _render_viruses(self):
        for virus in self.viruses:
            x, y = int(virus['x']), int(virus['y'])
            # Glitch effect
            for _ in range(3):
                offset_x = self.np_random.integers(-4, 5)
                offset_y = self.np_random.integers(-4, 5)
                size = self.np_random.integers(15, 25)
                rect = pygame.Rect(x + offset_x - size // 2, y + offset_y - size // 2, size, size)
                pygame.draw.rect(self.screen, self.COLOR_VIRUS, rect)

    def _render_player(self):
        x, y = int(self.player_x), int(self.PLAYER_Y_POS)
        radius = self.PLAYER_HITBOX_RADIUS

        # Glow
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius + 10, self.COLOR_PLAYER_GLOW)

        # Player triangle
        points = [
            (x, y - radius),
            (x - radius, y + radius // 2),
            (x + radius, y + radius // 2)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_surge(self):
        if self.surge_state['active']:
            progress = 1 - (self.surge_state['timer'] / self.SURGE_DURATION)
            current_radius = int(self.SURGE_MAX_RADIUS * progress)
            alpha = int(200 * (1 - progress))
            if current_radius > 0 and alpha > 0:
                color = (*self.COLOR_SURGE, alpha)
                pygame.gfxdraw.aacircle(self.screen, int(self.player_x), int(self.PLAYER_Y_POS), current_radius, color)

        # Cooldown indicator
        if self.surge_state['cooldown'] > 0:
            cooldown_progress = self.surge_state['cooldown'] / (self.SURGE_COOLDOWN + self.SURGE_DURATION)
            bar_width = 50
            bar_height = 5
            fill_width = int(bar_width * cooldown_progress)

            bar_x = self.player_x - bar_width / 2
            bar_y = self.PLAYER_Y_POS + self.PLAYER_HITBOX_RADIUS + 10

            pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_SURGE, (bar_x, bar_y, fill_width, bar_height))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = (*p['color'], alpha)
            if p['size'] > 0:
                surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(surf, (p['x'] - p['size'], p['y'] - p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        health_text = self.font_ui.render(f"HEALTH: {self.health}", True, self.COLOR_UI_TEXT)
        health_rect = health_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(health_text, health_rect)

        # Skill Points
        skill_text = self.font_ui.render(f"SKILL PTS: {self.skill_points}", True, self.COLOR_UI_TEXT)
        skill_rect = skill_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 40))
        self.screen.blit(skill_text, skill_rect)

        # Combo
        if self.combo > 1:
            # Make text pulse
            scale = 1 + 0.1 * math.sin(self.steps * 0.5)
            scaled_font = pygame.font.Font(None, int(48 * scale))
            combo_text = scaled_font.render(f"x{self.combo}", True, self.COLOR_COMBO_TEXT)
            combo_rect = combo_text.get_rect(bottomright=(self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 20))
            self.screen.blit(combo_text, combo_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            game_over_text = self.font_game_over.render("CONNECTION LOST", True, self.COLOR_VIRUS)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(game_over_text, text_rect)

            final_score_text = self.font_ui.render(f"FINAL SCORE: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 40))
            self.screen.blit(final_score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.health,
            "combo": self.combo,
            "skill_points": self.skill_points
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will create a window and display the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Data Stream Surge")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0

    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0  # None
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_pressed, shift_pressed]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(env.metadata['render_fps'])

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
    env.close()