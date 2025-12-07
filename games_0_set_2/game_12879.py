import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:35:50.356515
# Source Brief: brief_02879.md
# Brief Index: 2879
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Chroma Maze'.

    The player controls an oscillating light beam, navigating a field of obstacles.
    The beam's color can be changed to interact with obstacles in different ways:
    - RED: Burns obstacles at high intensity.
    - GREEN: Passes through obstacles harmlessly.
    - BLUE: Slows obstacles, preventing a game-over on collision.

    The goal is to collect 50 light particles to win.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control an oscillating light beam and navigate a field of obstacles. "
        "Cycle the beam's color to burn, pass through, or slow obstacles while collecting light particles."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the beam. Press space to cycle the beam's color (Red, Green, Blue)."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_PARTICLE_COUNT = 50
    MAX_STEPS = 2500
    INITIAL_OBSTACLES = 10
    PARTICLES_ON_SCREEN = 7
    BEAM_SPEED = 4.0
    INTENSITY_PERIOD_STEPS = 90  # 3 seconds at 30 FPS

    # --- COLORS ---
    COLOR_BG = (15, 10, 40)
    COLOR_WALL = (40, 30, 80)
    COLOR_BEAM_RED = (255, 50, 50)
    COLOR_BEAM_GREEN = (50, 255, 50)
    COLOR_BEAM_BLUE = (50, 100, 255)
    COLOR_PARTICLE = (255, 255, 100)
    COLOR_OBSTACLE = (140, 80, 200)
    COLOR_OBSTACLE_SLOWED = (180, 150, 220)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_BG = (0, 0, 0, 128)

    BEAM_COLORS = [COLOR_BEAM_RED, COLOR_BEAM_GREEN, COLOR_BEAM_BLUE]
    BEAM_RED, BEAM_GREEN, BEAM_BLUE = 0, 1, 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- GYMNASIUM SPACES ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- GAME STATE ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.beam_pos = np.array([0.0, 0.0])
        self.beam_color_idx = 0
        self.beam_intensity = 0.0
        self.last_space_held = False

        self.obstacles = []
        self.particles = []
        self.effects = []
        self.particles_collected = 0
        self.obstacle_target_count = 0
        self.last_dist_to_particle = float('inf')
        
        # This call is for development; validates the implementation against the brief.
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- RESET GAME STATE ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles_collected = 0
        
        self.beam_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.beam_color_idx = self.BEAM_RED
        self.last_space_held = False

        self.obstacles.clear()
        self.particles.clear()
        self.effects.clear()

        self.obstacle_target_count = self.INITIAL_OBSTACLES
        for _ in range(self.obstacle_target_count):
            self._spawn_obstacle()

        for _ in range(self.PARTICLES_ON_SCREEN):
            self._spawn_particle()
            
        self.last_dist_to_particle = self._get_dist_to_nearest_particle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is already over, do nothing and return the final state
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- HANDLE ACTIONS ---
        self._handle_input(action)

        # --- UPDATE GAME LOGIC ---
        self._update_beam_intensity()
        
        # Distance reward calculation
        dist_to_particle = self._get_dist_to_nearest_particle()
        if dist_to_particle < self.last_dist_to_particle:
            reward += 0.1 # Closer to particle
        else:
            reward -= 0.1 # Further from particle
        self.last_dist_to_particle = dist_to_particle
        
        reward += self._check_collisions()
        self._update_effects()
        self._update_obstacle_count()

        # --- CHECK TERMINATION ---
        terminated = False
        truncated = False
        if self.particles_collected >= self.WIN_PARTICLE_COUNT:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100.0  # Win reward
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True # Use truncated for time limit
            reward -= 100.0  # Timeout penalty
        elif self.game_over: # Loss by collision
            terminated = True
            reward -= 100.0 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held_raw, _ = action
        space_held = space_held_raw == 1

        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1:  # Up
            move_vec[1] -= 1
        elif movement == 2:  # Down
            move_vec[1] += 1
        elif movement == 3:  # Left
            move_vec[0] -= 1
        elif movement == 4:  # Right
            move_vec[0] += 1
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        
        self.beam_pos += move_vec * self.BEAM_SPEED
        
        # Clamp position to screen bounds
        self.beam_pos[0] = np.clip(self.beam_pos[0], 0, self.SCREEN_WIDTH)
        self.beam_pos[1] = np.clip(self.beam_pos[1], 0, self.SCREEN_HEIGHT)

        # Color cycling (on press, not hold)
        if space_held and not self.last_space_held:
            self.beam_color_idx = (self.beam_color_idx + 1) % 3
            # SFX: color_change.wav
        self.last_space_held = space_held

    def _update_beam_intensity(self):
        # Oscillates between 0.0 and 1.0 over the period
        self.beam_intensity = (math.sin(self.steps / self.INTENSITY_PERIOD_STEPS * 2 * math.pi) + 1) / 2

    def _update_obstacle_count(self):
        # "Obstacle count increases by 1 every 2 collected particles after the initial 10"
        if self.particles_collected > 0:
            self.obstacle_target_count = self.INITIAL_OBSTACLES + (self.particles_collected // 2)
        
        while len(self.obstacles) < self.obstacle_target_count:
            self._spawn_obstacle()

    def _check_collisions(self):
        reward = 0
        beam_rect = pygame.Rect(self.beam_pos[0] - 2, self.beam_pos[1] - 2, 4, 4)

        # Particle collisions
        for particle in self.particles[:]:
            if np.linalg.norm(self.beam_pos - particle['pos']) < 10:
                self.particles.remove(particle)
                self.particles_collected += 1
                self.score += 1
                reward += 1.0
                self._spawn_particle()
                # SFX: collect_particle.wav
                break # Only collect one per frame

        # Obstacle collisions
        for obstacle in self.obstacles[:]:
            if obstacle['rect'].colliderect(beam_rect):
                if self.beam_color_idx == self.BEAM_GREEN:
                    # Pass through
                    continue
                elif self.beam_color_idx == self.BEAM_RED and self.beam_intensity > 0.85:
                    # Burn obstacle
                    self._create_particle_explosion(obstacle['rect'].center, self.COLOR_OBSTACLE)
                    self.obstacles.remove(obstacle)
                    reward += 0.5 # Small reward for clearing obstacle
                    # SFX: burn_obstacle.wav
                elif self.beam_color_idx == self.BEAM_BLUE:
                    # Slow obstacle
                    if obstacle['state'] == 'normal':
                        obstacle['state'] = 'slowed'
                        reward += 0.2 # Small reward for neutralizing threat
                        # SFX: slow_obstacle.wav
                elif obstacle['state'] == 'normal':
                    # Lethal collision
                    self.game_over = True
                    reward -= 5.0 # Collision penalty
                    self._create_particle_explosion(self.beam_pos, self.BEAM_COLORS[self.beam_color_idx])
                    # SFX: player_death.wav
        return reward

    def _get_dist_to_nearest_particle(self):
        if not self.particles:
            return float('inf')
        
        particle_positions = np.array([p['pos'] for p in self.particles])
        distances = np.linalg.norm(particle_positions - self.beam_pos, axis=1)
        return np.min(distances)

    def _spawn_entity(self, size):
        max_tries = 100
        for _ in range(max_tries):
            pos = pygame.Rect(
                self.np_random.integers(20, self.SCREEN_WIDTH - 20 - size[0]),
                self.np_random.integers(20, self.SCREEN_HEIGHT - 20 - size[1]),
                *size
            )
            # Check for overlap with player start, other obstacles, and particles
            if pos.colliderect(pygame.Rect(self.SCREEN_WIDTH/2 - 30, self.SCREEN_HEIGHT/2 - 30, 60, 60)):
                continue
            
            is_overlapping = False
            for obs in self.obstacles:
                if pos.colliderect(obs['rect']):
                    is_overlapping = True
                    break
            if is_overlapping: continue

            for part in self.particles:
                if pos.collidepoint(part['pos']):
                    is_overlapping = True
                    break
            if is_overlapping: continue
            
            return pos
        return None # Failed to spawn

    def _spawn_obstacle(self):
        size = (self.np_random.integers(30, 60), self.np_random.integers(30, 60))
        rect = self._spawn_entity(size)
        if rect:
            self.obstacles.append({'rect': rect, 'state': 'normal'})

    def _spawn_particle(self):
        pos_rect = self._spawn_entity((10, 10))
        if pos_rect:
            self.particles.append({'pos': np.array(pos_rect.center, dtype=np.float64)})

    def _create_particle_explosion(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.effects.append({
                'pos': np.array(pos, dtype=np.float64),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_effects(self):
        for effect in self.effects[:]:
            effect['pos'] += effect['vel']
            effect['vel'] *= 0.95 # friction
            effect['life'] -= 1
            if effect['life'] <= 0:
                self.effects.remove(effect)

    def _get_observation(self):
        # --- RENDER ALL ELEMENTS TO THE PYGAME SURFACE ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # --- CONVERT TO NUMPY ARRAY ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for particle in self.particles:
            pos = (int(particle['pos'][0]), int(particle['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_PARTICLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_PARTICLE)

        # Render obstacles
        for obstacle in self.obstacles:
            color = self.COLOR_OBSTACLE_SLOWED if obstacle['state'] == 'slowed' else self.COLOR_OBSTACLE
            pygame.draw.rect(self.screen, color, obstacle['rect'], border_radius=3)

        # Render beam (if not game over from collision)
        if not (self.game_over and not self.win):
            self._draw_glowing_circle(
                self.screen,
                self.beam_pos,
                10 + 15 * self.beam_intensity,
                self.BEAM_COLORS[self.beam_color_idx]
            )

        # Render effects
        for effect in self.effects:
            pos = (int(effect['pos'][0]), int(effect['pos'][1]))
            size = int(effect['life'] / 10)
            if size > 0:
                pygame.draw.rect(self.screen, effect['color'], (pos[0], pos[1], size, size))

    def _draw_glowing_circle(self, surface, pos, radius, color):
        pos_int = (int(pos[0]), int(pos[1]))
        base_radius = int(radius * 0.3)
        
        # Draw concentric circles for glow effect
        for i in range(int(radius - base_radius), 0, -2):
            alpha = 40 * (1 - (i / (radius - base_radius)))
            glow_color = (*color, int(alpha))
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], base_radius + i, glow_color)
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], base_radius + i, glow_color)

        # Draw solid core
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], base_radius, color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], base_radius, color)


    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Particles collected
        particle_text = f"Particles: {self.particles_collected}/{self.WIN_PARTICLE_COUNT}"
        text_surf = self.font_main.render(particle_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Score
        score_text = f"Score: {int(self.score)}"
        text_surf = self.font_main.render(score_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)
        
        # Beam Intensity Bar
        intensity_rect_bg = pygame.Rect(self.SCREEN_WIDTH / 2 - 50, 15, 100, 10)
        intensity_rect_fg = pygame.Rect(self.SCREEN_WIDTH / 2 - 50, 15, 100 * self.beam_intensity, 10)
        pygame.draw.rect(self.screen, (50,50,50), intensity_rect_bg, border_radius=3)
        pygame.draw.rect(self.screen, self.BEAM_COLORS[self.beam_color_idx], intensity_rect_fg, border_radius=3)

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BEAM_GREEN if self.win else self.COLOR_BEAM_RED
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a translucent overlay
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "particles_collected": self.particles_collected,
            "beam_intensity": self.beam_intensity,
            "beam_color": ["RED", "GREEN", "BLUE"][self.beam_color_idx]
        }
    
    def render(self):
        # This method is not required by the core gym API but is good practice
        # It's used for human playback.
        # This implementation uses the rgb_array from _get_observation.
        return self._get_observation()
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # --- HUMAN PLAYBACK ---
    # Un-comment the line below to run with display
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chroma Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.metadata["render_fps"])

    env.close()