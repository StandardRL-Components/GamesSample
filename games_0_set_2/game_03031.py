import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A procedurally generated side-view rhythm runner where players jump over obstacles to the beat.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press Space to jump. Time your jumps with the beat for bonus points and to build your combo."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, neon-infused rhythm runner. Jump over geometric obstacles to the beat to maximize your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.BPM = 100.0
        self.FRAMES_PER_BEAT = int(self.FPS * 60 / self.BPM)
        
        # Colors
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_GRID = (40, 30, 80)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 255)
        self.OBSTACLE_COLORS = [(255, 0, 128), (0, 255, 255), (255, 255, 0)]
        
        # Player Physics
        self.PLAYER_X = self.SCREEN_WIDTH // 5
        self.PLAYER_SIZE = 20
        self.GROUND_Y = self.SCREEN_HEIGHT - 80
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -14

        # Game Rules
        self.MAX_LIVES = 3
        self.MAX_STEPS = 2000
        self.INITIAL_OBSTACLE_SPEED = 5.0
        self.BEAT_PERFECT_WINDOW = 2 # frames

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_combo = pygame.font.Font(None, 32)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.combo = 0
        self.combo_pop = 0
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = True
        self.last_space_held = False
        self.beat_timer = 0
        self.obstacle_speed = 0.0
        self.obstacles = []
        self.particles = []
        self.next_obstacle_spawn_dist = 0
        self.rng = None
        
        # This will be properly initialized in reset()
        self.reset(seed=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        elif self.rng is None:
            # Initialize RNG on first reset if no seed is provided
            self.rng = random.Random()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.MAX_LIVES
        self.combo = 1
        self.combo_pop = 0
        
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.on_ground = True
        
        self.last_space_held = False
        self.beat_timer = 0
        
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacles = []
        self.particles = []
        
        self._spawn_initial_obstacles()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- Action Handling ---
        space_held = action[1] == 1
        jump_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        # --- Game Logic Update ---
        self.steps += 1
        self.beat_timer = (self.beat_timer + 1) % self.FRAMES_PER_BEAT
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed += 0.05
        
        # Player update
        self._update_player(jump_pressed)
        
        # Obstacles update
        self._update_obstacles()
        
        # Particles update
        self._update_particles()
        
        # --- Collision, Scoring, and Reward Calculation ---
        step_reward, hit_obstacle = self._handle_collisions_and_scoring()
        reward += step_reward
        
        if hit_obstacle:
            self.lives -= 1
            self.combo = 1
            reward -= 5.0
            # sfx: player_hit
            self._create_particles(self.PLAYER_X + self.PLAYER_SIZE / 2, self.player_y - self.PLAYER_SIZE / 2, 30, (255, 80, 80))
        
        if jump_pressed and self.on_ground:
            # sfx: jump
            self._create_particles(self.PLAYER_X + self.PLAYER_SIZE / 2, self.GROUND_Y, 10, self.COLOR_PLAYER)
            is_on_beat = self.beat_timer <= self.BEAT_PERFECT_WINDOW or self.beat_timer >= self.FRAMES_PER_BEAT - self.BEAT_PERFECT_WINDOW
            if is_on_beat:
                reward += 2.0
                # sfx: perfect_jump
                self._create_particles(self.PLAYER_X + self.PLAYER_SIZE / 2, self.GROUND_Y, 20, (255, 255, 0), is_beat_pulse=True)

        # Survival reward
        reward += 0.1

        # --- Termination Check ---
        terminated = self.lives <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.steps >= self.MAX_STEPS:
                reward += 100.0 # Victory reward
                # sfx: level_complete
            else:
                reward -= 0 # No extra penalty on game over, hit penalty is enough
                # sfx: game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, jump_pressed):
        was_on_ground = self.on_ground
        
        if jump_pressed and self.on_ground:
            self.player_vy = self.JUMP_STRENGTH
            self.on_ground = False
        
        if not self.on_ground:
            self.player_vy += self.GRAVITY
            self.player_y += self.player_vy
        
        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            self.on_ground = True
            if not was_on_ground:
                # sfx: land
                self._create_particles(self.PLAYER_X + self.PLAYER_SIZE / 2, self.GROUND_Y, 5, self.COLOR_PLAYER)

    def _update_obstacles(self):
        # Move existing obstacles and remove off-screen ones
        self.obstacles = [ob for ob in self.obstacles if ob['x'] + ob['w'] > 0]
        for ob in self.obstacles:
            ob['x'] -= self.obstacle_speed
            
        # Spawn new obstacles
        if not self.obstacles or self.obstacles[-1]['x'] < self.SCREEN_WIDTH - self.next_obstacle_spawn_dist:
            self._spawn_obstacle()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _handle_collisions_and_scoring(self):
        reward = 0.0
        hit_obstacle = False
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for ob in self.obstacles:
            # Check for passing
            if not ob['passed'] and ob['x'] + ob['w'] < self.PLAYER_X:
                ob['passed'] = True
                self.score += 10 * self.combo
                reward += 1.0
                self.combo += 1
                self.combo_pop = 15 # frames for pop animation
                # sfx: score_point
            
            # Check for collision
            if not ob['hit']:
                ob_rect = pygame.Rect(ob['x'], ob['y'] - ob['h'], ob['w'], ob['h'])
                if player_rect.colliderect(ob_rect):
                    ob['hit'] = True
                    hit_obstacle = True

        return reward, hit_obstacle

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "combo": self.combo,
        }

    def _render_background(self):
        # Ground line
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 3)

        # Pulsating grid
        beat_progress = self.beat_timer / self.FRAMES_PER_BEAT
        pulse = (1 - abs(0.5 - beat_progress) * 2) ** 2 # Parabolic pulse
        
        # Horizontal lines
        for i in range(1, 10):
            y = self.GROUND_Y - i * 40
            if y < 0: break
            alpha = int(30 + pulse * 40 * (1 - i / 10))
            color = (*self.COLOR_GRID, alpha)
            line_surf = pygame.Surface((self.SCREEN_WIDTH, 1), pygame.SRCALPHA)
            pygame.draw.line(line_surf, color, (0, 0), (self.SCREEN_WIDTH, 0))
            self.screen.blit(line_surf, (0, y))
            
        # Vertical lines
        offset = (self.steps * self.obstacle_speed) % 40
        for i in range(self.SCREEN_WIDTH // 40 + 2):
            x = i * 40 - offset
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GROUND_Y), 1)

    def _render_obstacles(self):
        for ob in self.obstacles:
            rect = (int(ob['x']), int(ob['y'] - ob['h']), int(ob['w']), int(ob['h']))
            pygame.gfxdraw.box(self.screen, rect, ob['color'])
            # Outline for better visibility
            outline_color = tuple(min(255, c + 50) for c in ob['color'])
            pygame.gfxdraw.rectangle(self.screen, rect, outline_color)

    def _render_player(self):
        player_rect = pygame.Rect(self.PLAYER_X, int(self.player_y - self.PLAYER_SIZE), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow effect
        glow_size = self.PLAYER_SIZE + 10
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        glow_alpha = 60 + 40 * math.sin(self.steps * 0.2)
        pygame.gfxdraw.box(glow_surf, (0,0,glow_size,glow_size), (*self.COLOR_PLAYER, int(glow_alpha)))
        self.screen.blit(glow_surf, (player_rect.x - 5, player_rect.y - 5), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.box(self.screen, player_rect, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                # Create a temporary surface for each particle to handle alpha blending correctly
                particle_surf = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
                color = (*p['color'], alpha)
                pygame.gfxdraw.box(particle_surf, (0, 0, p['size'], p['size']), color)
                self.screen.blit(particle_surf, (int(p['x'] - p['size'] / 2), int(p['y'] - p['size'] / 2)))


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(life_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.lives):
            rect = (self.SCREEN_WIDTH - 80 + i * (self.PLAYER_SIZE // 2 + 5), 12, self.PLAYER_SIZE // 2, self.PLAYER_SIZE // 2)
            pygame.gfxdraw.box(self.screen, rect, self.COLOR_PLAYER)

        # Combo
        if self.combo > 1:
            if self.combo_pop > 0:
                self.combo_pop -= 1
                pop_scale = 1.0 + 0.5 * (self.combo_pop / 15)
            else:
                pop_scale = 1.0

            font_size = int(self.font_combo.get_height() * pop_scale)
            temp_font = pygame.font.Font(None, font_size)
            combo_text = temp_font.render(f"x{self.combo}", True, (255, 255, 0))
            text_rect = combo_text.get_rect(center=(self.PLAYER_X + self.PLAYER_SIZE / 2, self.player_y - self.PLAYER_SIZE - 20))
            self.screen.blit(combo_text, text_rect)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "LEVEL COMPLETE" if self.steps >= self.MAX_STEPS else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"FINAL SCORE: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
            self.screen.blit(final_score_text, score_rect)

    def _spawn_obstacle(self):
        h = self.rng.choice([30, 60])
        w = self.rng.choice([20, 40])
        self.obstacles.append({
            'x': self.SCREEN_WIDTH,
            'y': self.GROUND_Y,
            'w': w,
            'h': h,
            'color': self.rng.choice(self.OBSTACLE_COLORS),
            'passed': False,
            'hit': False
        })
        self.next_obstacle_spawn_dist = self.rng.randint(int(self.PLAYER_SIZE * 6), int(self.PLAYER_SIZE * 12))

    def _spawn_initial_obstacles(self):
        current_x = self.SCREEN_WIDTH * 0.7
        while current_x < self.SCREEN_WIDTH * 1.5:
            h = self.rng.choice([30, 60])
            w = self.rng.choice([20, 40])
            self.obstacles.append({
                'x': current_x, 'y': self.GROUND_Y, 'w': w, 'h': h,
                'color': self.rng.choice(self.OBSTACLE_COLORS), 'passed': False, 'hit': False
            })
            current_x += self.rng.randint(int(self.PLAYER_SIZE * 8), int(self.PLAYER_SIZE * 15))
        self.next_obstacle_spawn_dist = self.rng.randint(int(self.PLAYER_SIZE * 6), int(self.PLAYER_SIZE * 12))

    def _create_particles(self, x, y, count, color, is_beat_pulse=False):
        for _ in range(count):
            if is_beat_pulse:
                angle = self.rng.uniform(0, 2 * math.pi)
                speed = self.rng.uniform(2, 5)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.rng.randint(20, 40)
            else:
                vx = self.rng.uniform(-2, 2)
                vy = self.rng.uniform(-5, 0)
                life = self.rng.randint(10, 25)
                
            self.particles.append({
                'x': x, 'y': y, 'vx': vx, 'vy': vy, 'size': self.rng.randint(3, 7),
                'life': life, 'max_life': life, 'color': color
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a graphical display. If you are in a headless environment,
    # comment out this block or run with a virtual display.
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = np.array([0, 0, 0])
    
    # --- Pygame window for human play ---
    pygame.display.set_caption("Rhythm Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    print("--- Playing Game ---")
    print(env.user_guide)

    while not done:
        # --- Handle player input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_r: # Reset on 'R'
                    obs, info = env.reset()
                    action.fill(0)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render the observation to the display ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # and surfarray expects (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Control frame rate ---
        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()