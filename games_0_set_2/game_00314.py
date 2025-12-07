import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode, essential for server-side execution
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Press SPACE to jump. Time your jumps with the beat to score points."
    )

    game_description = (
        "A fast-paced rhythm platformer. Navigate a procedurally generated obstacle course "
        "by jumping to the beat, maximizing your score through precise timing."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    COURSE_LENGTH = 2000  # Max steps

    # Colors
    COLOR_BG_TOP = pygame.Color("#2c003e")
    COLOR_BG_BOTTOM = pygame.Color("#001f3f")
    COLOR_GROUND = pygame.Color("#3d2556")
    COLOR_PLAYER = pygame.Color("#00ffff")
    COLOR_PLAYER_GLOW = pygame.Color("#aaffff")
    COLOR_OBSTACLE = pygame.Color("#ff4136")
    COLOR_OBSTACLE_GLOW = pygame.Color("#ff8c87")
    COLOR_RHYTHM_CUE = pygame.Color("#01ff70")
    COLOR_TEXT = pygame.Color("#ffffff")
    COLOR_PARTICLE_JUMP = pygame.Color("#aaffff")
    COLOR_PARTICLE_HIT = pygame.Color("#ff8c87")

    # Player settings
    PLAYER_X = 100
    PLAYER_WIDTH = 25
    PLAYER_HEIGHT = 25
    GROUND_Y = 350
    GRAVITY = 0.8
    JUMP_STRENGTH = -15

    # Game settings
    INITIAL_LIVES = 3
    INITIAL_OBSTACLE_SPEED = 4.0
    INITIAL_OBSTACLE_GAP_BEATS = 3.0
    BEATS_PER_MINUTE = 120
    PERFECT_TIMING_WINDOW = 3  # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.beat_period_frames = int(60 / self.BEATS_PER_MINUTE * self.FPS)

        self.prev_space_held = False
        self.particles = []

        self.background_surface = self._create_gradient_background()

        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES

        self.player_y = self.GROUND_Y - self.PLAYER_HEIGHT
        self.player_vy = 0
        self.is_jumping = False

        self.obstacles = []
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_spawn_gap_beats = self.INITIAL_OBSTACLE_GAP_BEATS
        self.next_obstacle_spawn_step = self.beat_period_frames * 4

        self.prev_space_held = False
        self.particles.clear()

        self.last_jump_timing_reward = 0
        self.last_collision_reward = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        # movement = action[0] # Not used in this game logic
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Not used in this game logic

        jump_triggered = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        reward = 0.1  # Survival reward

        # --- Player Logic ---
        if jump_triggered and not self.is_jumping:
            self.player_vy = self.JUMP_STRENGTH
            self.is_jumping = True
            self._spawn_particles(20, (self.PLAYER_X + self.PLAYER_WIDTH / 2, self.GROUND_Y), self.COLOR_PARTICLE_JUMP)

            # Check jump timing for reward
            frames_off_beat = self.steps % self.beat_period_frames
            if frames_off_beat <= self.PERFECT_TIMING_WINDOW or \
               self.beat_period_frames - frames_off_beat <= self.PERFECT_TIMING_WINDOW:
                reward += 1.0
                self.score += 100
                self.last_jump_timing_reward = 20  # for visual feedback
            else:
                self.score += 10

        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy

        if self.player_y >= self.GROUND_Y - self.PLAYER_HEIGHT:
            self.player_y = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_vy = 0
            if self.is_jumping:  # Landed
                self.is_jumping = False

        # --- Obstacle Logic ---
        self.steps += 1

        # Spawn new obstacles
        if self.steps >= self.next_obstacle_spawn_step and self.steps < self.COURSE_LENGTH - 150:
            obstacle_height = self.np_random.integers(30, 80)
            self.obstacles.append(pygame.Rect(
                self.WIDTH,
                self.GROUND_Y - obstacle_height,
                self.np_random.integers(20, 40),
                obstacle_height
            ))
            spawn_delay_frames = int(self.obstacle_spawn_gap_beats * self.beat_period_frames)
            self.next_obstacle_spawn_step = self.steps + spawn_delay_frames

        # Move existing obstacles
        player_rect = self._get_player_rect()
        collided_this_frame = False
        for obs in self.obstacles[:]:
            obs.x -= self.obstacle_speed
            if obs.right < 0:
                self.obstacles.remove(obs)
            elif player_rect.colliderect(obs):
                self.obstacles.remove(obs)
                collided_this_frame = True

        if collided_this_frame:
            self.lives -= 1
            reward -= 5.0
            self.score -= 500
            self._spawn_particles(30, player_rect.center, self.COLOR_PARTICLE_HIT)
            self.last_collision_reward = 20  # for visual feedback

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed += 0.2
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_spawn_gap_beats = max(1.0, self.obstacle_spawn_gap_beats - 0.1)

        # --- Termination ---
        terminated = self.lives <= 0 or self.steps >= self.COURSE_LENGTH
        if terminated and not self.game_over:
            if self.lives > 0:  # Reached the end
                reward += 50.0
                self.score += 5000
            self.game_over = True

        # --- Visual Feedback Timers ---
        self.last_jump_timing_reward = max(0, self.last_jump_timing_reward - 1)
        self.last_collision_reward = max(0, self.last_collision_reward - 1)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_player_rect(self):
        return pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

    def _get_observation(self):
        # --- Drawing ---
        self.screen.blit(self.background_surface, (0, 0))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Rhythm Cue
        self._render_rhythm_cue()

        # Obstacles
        for obs in self.obstacles:
            self._draw_glowing_rect(self.screen, obs, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)

        # Player
        player_rect = self._get_player_rect()
        self._draw_glowing_rect(self.screen, player_rect, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

        # Particles
        self._update_and_draw_particles()

        # UI
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = self.COLOR_BG_TOP.lerp(self.COLOR_BG_BOTTOM, ratio)
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_rhythm_cue(self):
        # Bottom pulsating circle
        pulse = (math.sin(self.steps / self.beat_period_frames * math.pi * 2) + 1) / 2  # 0 to 1
        pulse = 1.0 - (1.0 - pulse) ** 2  # Ease out

        radius = int(20 + pulse * 15)
        alpha = int(50 + pulse * 150)
        center = (self.WIDTH // 2, self.GROUND_Y + 25)

        color_with_alpha = (self.COLOR_RHYTHM_CUE.r, self.COLOR_RHYTHM_CUE.g, self.COLOR_RHYTHM_CUE.b, alpha)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color_with_alpha)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color_with_alpha)

        # Vertical beat lines on ground
        for i in range(-2, 10):
            beat_step = (self.steps // self.beat_period_frames + i) * self.beat_period_frames
            x_pos = self.WIDTH - (beat_step - self.steps) * self.obstacle_speed
            if 0 < x_pos < self.WIDTH:
                pygame.draw.line(self.screen, self.COLOR_GROUND.lerp(self.COLOR_RHYTHM_CUE, 0.5), (x_pos, self.GROUND_Y), (x_pos, self.GROUND_Y + 5), 2)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Visual feedback for events
        if self.last_jump_timing_reward > 0:
            perfect_text = self.font_ui.render("PERFECT!", True, self.COLOR_RHYTHM_CUE)
            pos = (self.PLAYER_X + 40, self.player_y - 20)
            self.screen.blit(perfect_text, pos)

        if self.last_collision_reward > 0:
            hit_text = self.font_ui.render("HIT!", True, self.COLOR_OBSTACLE)
            pos = (self.PLAYER_X + 40, self.player_y - 20)
            self.screen.blit(hit_text, pos)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        end_message = "COURSE COMPLETE!" if self.lives > 0 else "GAME OVER"
        text_surface = self.font_game_over.render(end_message, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text_surface, text_rect)

    def _draw_glowing_rect(self, surface, rect, color, glow_color):
        # Draw blurred glow effect by drawing larger rects with low alpha
        for i in range(5, 0, -1):
            glow_rect = rect.inflate(i * 4, i * 4)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            alpha = 40 - i * 7
            pygame.draw.rect(glow_surface, (glow_color.r, glow_color.g, glow_color.b, alpha), (0, 0, *glow_rect.size), border_radius=5)
            surface.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=3)
        pygame.draw.rect(surface, glow_color, rect, width=2, border_radius=3)

    def _spawn_particles(self, count, pos, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                radius = int(4 * (p['life'] / p['max_life']))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(
                        self.screen, int(p['pos'][0]), int(p['pos'][1]),
                        radius, (p['color'].r, p['color'].g, p['color'].b, alpha)
                    )

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    # It will not work in a headless environment
    # Re-enable the display driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Rhythm Platformer")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        action = [0, 0, 0]  # no-op
        if keys[pygame.K_SPACE]:
            action[1] = 1  # space held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False
                total_reward = 0

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        # Render the observation from the environment
        # The observation is (H, W, C), but pygame blit needs a surface
        # or an array in (W, H, C) format. So we transpose back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()