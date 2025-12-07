
# Generated: 2025-08-28T03:48:39.311740
# Source Brief: brief_02133.md
# Brief Index: 2133

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space to jump. Time your jumps to the beat to build a combo!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling rhythm game. Jump over procedurally generated obstacles to a dynamic beat."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    BPM = 120
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_START = (20, 10, 40)
    COLOR_BG_END = (60, 30, 80)
    COLOR_GROUND = (100, 100, 110)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255, 50)
    COLOR_OBSTACLE = (255, 80, 80)
    COLOR_PARTICLE_GOOD = (80, 255, 80)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BEAT_INDICATOR = (200, 200, 255)

    # Player Physics
    PLAYER_X = 120
    PLAYER_SIZE = 20
    GROUND_Y = 350
    GRAVITY = 0.9
    JUMP_STRENGTH = -15
    JUMP_HOLD_PENALTY_THRESHOLD = 10 # frames

    # Obstacle Mechanics
    OBSTACLE_WIDTH = 30
    OBSTACLE_INITIAL_SPEED = 6.0
    OBSTACLE_SPEED_INCREASE = 0.05
    INITIAL_SPAWN_FRAMES = 75 # 2.5 seconds
    SPAWN_RATE_INCREASE = 0.99 # 1% faster

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_combo = pygame.font.SysFont("Consolas", 32, bold=True)

        self.frames_per_beat = (60 / self.BPM) * self.FPS

        self.bg_surface = self._create_gradient_background()

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel_y = 0.0
        self.is_jumping = False
        self.jump_held_frames = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.combo = 1
        self.game_over = False
        self.beat_timer = 0
        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_rate = self.INITIAL_SPAWN_FRAMES
        self.obstacle_speed = self.OBSTACLE_INITIAL_SPEED
        self.cleared_obstacles_count = 0

        self.reset()
        
        # This can be commented out for performance after validation
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.PLAYER_X, self.GROUND_Y)
        self.player_vel_y = 0.0
        self.is_jumping = False
        self.jump_held_frames = 0
        
        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.combo = 1
        self.game_over = False
        
        self.beat_timer = 0
        self.obstacle_spawn_timer = self.INITIAL_SPAWN_FRAMES
        self.obstacle_spawn_rate = self.INITIAL_SPAWN_FRAMES
        self.obstacle_speed = self.OBSTACLE_INITIAL_SPEED
        self.cleared_obstacles_count = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        space_held = action[1] == 1

        reward = 0.0
        
        # --- Update Game State ---
        self._handle_input(space_held)
        self._update_player()
        self._update_obstacles()
        self._update_particles()
        self._update_difficulty()
        self.beat_timer = (self.beat_timer + 1) % self.frames_per_beat
        self.steps += 1

        # --- Collision and Termination Check ---
        player_rect = self._get_player_rect()
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                self.game_over = True
                # Sound: Player hit
                break
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        # --- Reward Calculation ---
        # Survival reward
        reward += 0.1

        # Penalty for holding jump too long
        if self.is_jumping and space_held and self.player_vel_y > 0:
            reward -= 0.2

        # Check for cleared obstacles and combo
        newly_cleared_count = 0
        player_left_edge = player_rect.left
        for obs in self.obstacles:
            if not obs['cleared'] and obs['rect'].right < player_left_edge:
                obs['cleared'] = True
                newly_cleared_count += 1
                # Sound: Obstacle clear
        
        if newly_cleared_count > 0:
            self.cleared_obstacles_count += newly_cleared_count
            reward += 1.0 * newly_cleared_count

        # Reset combo if on ground and an obstacle passes
        if not self.is_jumping:
            for obs in self.obstacles:
                if obs['rect'].right < self.player_pos.x and not obs['cleared']:
                    if self.combo > 1:
                        # Sound: Combo break
                        pass
                    self.combo = 1
                    break

        # Terminal rewards
        if self.game_over:
            reward = -100.0
        elif self.steps >= self.MAX_STEPS:
            reward = 100.0
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, space_held):
        if space_held and not self.is_jumping:
            self.is_jumping = True
            self.player_vel_y = self.JUMP_STRENGTH
            self.jump_held_frames = 0
            # Sound: Jump
            self._spawn_particles(self.player_pos.x, self.GROUND_Y + 5, 10, self.COLOR_PLAYER)
        
        if space_held and self.is_jumping:
            self.jump_held_frames += 1

    def _update_player(self):
        self.player_vel_y += self.GRAVITY
        self.player_pos.y += self.player_vel_y

        if self.player_pos.y >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel_y = 0
            if self.is_jumping: # Just landed
                self.is_jumping = False
                # Sound: Land
                self._spawn_particles(self.player_pos.x, self.GROUND_Y + 5, 20, self.COLOR_GROUND)
                
                # Check for perfect timing
                beat_window = 2
                is_on_beat = self.beat_timer < beat_window or self.beat_timer > self.frames_per_beat - beat_window
                if is_on_beat:
                    self.combo += 1
                    self.score += 1.0 * self.combo # Event-based reward
                    # Sound: Perfect timing
                    self._spawn_particles(self.player_pos.x, self.GROUND_Y - 10, 30, self.COLOR_PARTICLE_GOOD)
                else:
                    if self.combo > 1:
                        # Sound: Combo break
                        pass
                    self.combo = 1

    def _update_obstacles(self):
        # Move existing obstacles
        for obs in self.obstacles:
            obs['rect'].x -= self.obstacle_speed
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

        # Spawn new obstacles
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            height = self.np_random.integers(40, 120)
            y_pos = self.GROUND_Y - height
            
            # Ensure a path is possible (cap height based on jump apex)
            max_jump_height = abs(self.JUMP_STRENGTH * 10) # rough estimate
            if height > max_jump_height:
                height = max_jump_height
                y_pos = self.GROUND_Y - height

            self.obstacles.append({
                'rect': pygame.Rect(self.WIDTH, y_pos, self.OBSTACLE_WIDTH, height),
                'cleared': False
            })
            self.obstacle_spawn_timer = self.obstacle_spawn_rate

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE
        if self.steps > 0 and self.steps % 100 == 0:
            self.obstacle_spawn_rate = max(20, self.obstacle_spawn_rate * self.SPAWN_RATE_INCREASE)

    def _get_player_rect(self):
        return pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE / 2,
            self.player_pos.y - self.PLAYER_SIZE,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )

    def _spawn_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 5)

        # Beat indicator
        beat_progress = self.beat_timer / self.frames_per_beat
        pulse = abs(math.sin(beat_progress * math.pi))
        pulse_radius = 10 + 10 * pulse
        pulse_alpha = 50 + 200 * pulse
        s = pygame.Surface((pulse_radius * 2, pulse_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_BEAT_INDICATOR, int(pulse_alpha)), (pulse_radius, pulse_radius), pulse_radius)
        self.screen.blit(s, (int(self.PLAYER_X - pulse_radius), int(self.GROUND_Y + 10)))

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
            # Add a slight highlight for 3D feel
            highlight_rect = obs['rect'].copy()
            highlight_rect.height = 4
            pygame.draw.rect(self.screen, (255, 150, 150), highlight_rect)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color_with_alpha = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color_with_alpha)

        # Player
        player_rect = self._get_player_rect()
        # Glow
        glow_radius = int(self.PLAYER_SIZE * 0.8 + 5 * abs(math.sin(self.steps * 0.1)))
        pygame.gfxdraw.filled_circle(self.screen, int(player_rect.centerx), int(player_rect.centery), glow_radius, self.COLOR_PLAYER_GLOW)
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        # Eye
        eye_x = player_rect.centerx + 3
        eye_y = player_rect.centery - 4
        pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, eye_y), 3)
        pygame.draw.circle(self.screen, (0, 0, 0), (eye_x + 1, eye_y), 1)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score):06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_ui.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        if self.combo > 1:
            combo_text = self.font_combo.render(f"x{self.combo}", True, self.COLOR_PARTICLE_GOOD)
            text_rect = combo_text.get_rect(center=(self.PLAYER_X, self.GROUND_Y - 80))
            self.screen.blit(combo_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "cleared_obstacles": self.cleared_obstacles_count,
        }

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp),
                int(self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp),
                int(self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp),
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override pygame screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Rhythm Jumper")
    
    terminated = False
    running = True
    total_reward = 0
    
    # --- Main Game Loop for Human Play ---
    while running:
        action = [0, 0, 0] # no-op, release space, release shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        else:
            # Display Game Over message
            font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
            text = font_game_over.render("GAME OVER", True, (255, 0, 0))
            text_rect = text.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2 - 30))
            env.screen.blit(text, text_rect)
            
            font_restart = pygame.font.SysFont("Consolas", 20)
            restart_text = font_restart.render("Press R to restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(GameEnv.WIDTH/2, GameEnv.HEIGHT/2 + 30))
            env.screen.blit(restart_text, restart_rect)

            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                total_reward = 0

        # Blit the observation to the display screen
        # Need to transpose back for pygame's display format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)

    env.close()