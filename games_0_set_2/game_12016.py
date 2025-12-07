import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:52:50.692051
# Source Brief: brief_02016.md
# Brief Index: 2016
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a cyberpunk-themed arcade game.
    The agent controls a robot to collect energy orbs, maintain energy,
    and trigger speed boosts to achieve a high score within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a cyberpunk robot to collect energy orbs and trigger speed boosts "
        "to achieve a high score before the timer runs out."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move your robot and collect the green orbs."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Assumed FPS for tuning game feel
    GAME_LOGIC_HZ = 10 # Game logic updates per second
    TIME_LIMIT_SECONDS = 120
    MAX_STEPS = TIME_LIMIT_SECONDS * GAME_LOGIC_HZ

    # Colors (Cyberpunk Neon)
    COLOR_BG = (10, 0, 30)
    COLOR_GRID = (30, 20, 70)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_ORB = (50, 255, 50)
    COLOR_ORB_GLOW = (50, 200, 50)
    COLOR_BOOST_TRAIL = (255, 0, 255)
    COLOR_TEXT = (220, 220, 255)
    COLOR_ENERGY_HIGH = (0, 255, 150)
    COLOR_ENERGY_MID = (255, 255, 0)
    COLOR_ENERGY_LOW = (255, 50, 50)

    # Game Parameters
    PLAYER_SIZE = 12
    PLAYER_SPEED = 8
    PLAYER_BOOST_SPEED_MULTIPLIER = 2.0
    ORB_SIZE = 8
    NUM_ORBS = 15
    ORB_COLLECTION_RADIUS = PLAYER_SIZE + ORB_SIZE
    ENERGY_PER_ORB = 15.0
    ENERGY_DECAY_RATE = 1.0 / GAME_LOGIC_HZ # 1% per second
    BOOST_ORB_REQUIREMENT = 5
    BOOST_TIME_WINDOW = 1.5 * GAME_LOGIC_HZ # 1.5 seconds
    BOOST_DURATION = 5.0 * GAME_LOGIC_HZ # 5 seconds
    WIN_SCORE = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # --- Game State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.orbs = None
        self.particles = None
        self.recent_collections = None
        
        self.steps = None
        self.score = None
        self.energy = None
        self.speed_boost_timer = None
        self.time_remaining = None
        self.game_over = None
        self.win_condition_met = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0, 0], dtype=np.float32)
        self.orbs = []
        self.particles = []
        self.recent_collections = []
        
        self.steps = 0
        self.score = 0
        self.energy = 100.0
        self.speed_boost_timer = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.win_condition_met = False
        
        for _ in range(self.NUM_ORBS):
            self._spawn_orb()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        self._handle_movement(movement)
        
        # --- Game Logic Updates ---
        self._update_player_position()
        self._update_timers_and_energy()
        self._update_particles()
        
        reward = 0
        
        # Low energy penalty
        if self.energy < 90.0:
            reward -= 0.01

        # Collision and collection
        collected_this_step = self._handle_orb_collection()
        if collected_this_step:
            reward += 0.1 * collected_this_step
            # SFX: Orb collection sound
            
            # Check for speed boost trigger
            if self._check_for_speed_boost():
                reward += 1.0
                self.speed_boost_timer = self.BOOST_DURATION
                # SFX: Powerup activation sound

        # --- Termination Check ---
        self.steps += 1
        self.time_remaining -= 1
        
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.win_condition_met = True
            self.game_over = True
        elif self.time_remaining <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_particles()
        self._render_orbs()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "time_remaining": self.time_remaining / self.GAME_LOGIC_HZ,
            "speed_boost_active": self.speed_boost_timer > 0,
        }

    # --- Helper Methods: Game Logic ---

    def _handle_movement(self, movement):
        target_vel = np.array([0, 0], dtype=np.float32)
        speed = self.PLAYER_SPEED
        if self.speed_boost_timer > 0:
            speed *= self.PLAYER_BOOST_SPEED_MULTIPLIER

        if movement == 1: # Up
            target_vel[1] = -speed
        elif movement == 2: # Down
            target_vel[1] = speed
        elif movement == 3: # Left
            target_vel[0] = -speed
        elif movement == 4: # Right
            target_vel[0] = speed
        
        # Smooth acceleration/deceleration
        self.player_vel = self.player_vel * 0.6 + target_vel * 0.4

    def _update_player_position(self):
        self.player_pos += self.player_vel / self.GAME_LOGIC_HZ
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

    def _update_timers_and_energy(self):
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= 1
            # Spawn boost trail particles
            if self.steps % 2 == 0:
                self._spawn_particles(1, self.player_pos, self.COLOR_BOOST_TRAIL, speed_mult=0.5, life=10)

        self.energy = max(0.0, self.energy - self.ENERGY_DECAY_RATE)
        if self.energy == 0.0:
            # Could add a penalty or negative effect for zero energy
            pass

    def _handle_orb_collection(self):
        collected_count = 0
        orbs_to_remove = []
        for i, orb in enumerate(self.orbs):
            dist = np.linalg.norm(self.player_pos - orb['pos'])
            if dist < self.ORB_COLLECTION_RADIUS:
                orbs_to_remove.append(i)
                self.score += 1
                collected_count += 1
                self.energy = min(100.0, self.energy + self.ENERGY_PER_ORB)
                self.recent_collections.append(self.steps)
                self._spawn_particles(20, orb['pos'], self.COLOR_ORB)
        
        if orbs_to_remove:
            # Remove from back to front to avoid index errors
            for i in sorted(orbs_to_remove, reverse=True):
                del self.orbs[i]
            for _ in range(len(orbs_to_remove)):
                self._spawn_orb()
        
        return collected_count

    def _check_for_speed_boost(self):
        # Prune old collection timestamps
        cutoff = self.steps - self.BOOST_TIME_WINDOW
        self.recent_collections = [t for t in self.recent_collections if t > cutoff]
        
        if len(self.recent_collections) >= self.BOOST_ORB_REQUIREMENT:
            self.recent_collections.clear() # Consume the collections for the boost
            return True
        return False

    def _spawn_orb(self):
        # Ensure orbs don't spawn on top of the player
        while True:
            pos = np.array([
                self.np_random.uniform(self.ORB_SIZE, self.SCREEN_WIDTH - self.ORB_SIZE),
                self.np_random.uniform(self.ORB_SIZE, self.SCREEN_HEIGHT - self.ORB_SIZE)
            ])
            if np.linalg.norm(pos - self.player_pos) > self.ORB_COLLECTION_RADIUS * 2:
                self.orbs.append({'pos': pos, 'anim_offset': self.np_random.uniform(0, 2 * math.pi)})
                break

    def _spawn_particles(self, count, pos, color, speed_mult=1.0, life=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'timer': self.np_random.integers(life // 2, life),
                'max_timer': life,
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['timer'] -= 1
        self.particles = [p for p in self.particles if p['timer'] > 0]

    # --- Helper Methods: Rendering ---

    def _draw_glow_circle(self, surface, color, glow_color, center, radius, glow_radius):
        center_int = (int(center[0]), int(center[1]))
        
        # Draw glow layers
        for i in range(glow_radius, 0, -2):
            alpha = int(70 * (1 - i / glow_radius))
            if alpha > 0:
                # Using gfxdraw for antialiasing is too slow for many particles
                # A simple filled circle is a good compromise
                pygame.draw.circle(surface, glow_color + (alpha,), center_int, radius + i, 0)
        
        # Draw main circle
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_player(self):
        pos = self.player_pos
        size = self.PLAYER_SIZE
        
        # Boost indicator
        if self.speed_boost_timer > 0:
            boost_alpha = min(255, int(255 * (self.speed_boost_timer / (self.BOOST_DURATION / 2))))
            s = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_BOOST_TRAIL + (boost_alpha,), (size*2, size*2), size*2)
            self.screen.blit(s, (int(pos[0] - size*2), int(pos[1] - size*2)))
        
        # Player body (a square)
        player_rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
        
        # Glow effect
        glow_surf = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW + (50,), (0, 0, size*4, size*4), border_radius=int(size))
        self.screen.blit(glow_surf, player_rect.topleft - np.array([size * 1.5, size * 1.5]))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_orbs(self):
        for orb in self.orbs:
            # Bobbing animation
            bob = math.sin(self.steps / 10.0 + orb['anim_offset']) * 2
            pos = orb['pos'] + np.array([0, bob])
            self._draw_glow_circle(self.screen, self.COLOR_ORB, self.COLOR_ORB_GLOW, pos, self.ORB_SIZE, 10)
    
    def _render_particles(self):
        for p in self.particles:
            life_percent = p['timer'] / p['max_timer']
            radius = int(self.ORB_SIZE / 2 * life_percent)
            if radius > 0:
                alpha = int(200 * life_percent)
                pygame.draw.circle(self.screen, p['color'] + (alpha,), (int(p['pos'][0]), int(p['pos'][1])), radius)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"ORBS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_str = f"TIME: {self.time_remaining / self.GAME_LOGIC_HZ:.1f}"
        timer_text = self.font_main.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # Energy Bar
        bar_width = 300
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        
        energy_percent = self.energy / 100.0
        current_bar_width = int(bar_width * energy_percent)
        
        if energy_percent > 0.6:
            bar_color = self.COLOR_ENERGY_HIGH
        elif energy_percent > 0.3:
            bar_color = self.COLOR_ENERGY_MID
        else:
            bar_color = self.COLOR_ENERGY_LOW
            # Low energy warning flash
            if self.steps % 10 < 5:
                bar_color = (255,255,255)

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), 2, border_radius=5)
        if current_bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, current_bar_width, bar_height), 0, border_radius=5)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition_met:
                end_text_str = "SYNCHRONIZATION COMPLETE"
                end_color = self.COLOR_PLAYER
            else:
                end_text_str = "CONNECTION LOST"
                end_color = self.COLOR_ENERGY_LOW
                
            end_text = self.font_main.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for rendering
    pygame.display.set_caption("Cyber Orb Sync")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Keyboard Controls ---
        keys = pygame.key.get_pressed()
        
        movement_action = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement_action, space_action, shift_action])
        
        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            # Wait a bit on the game over screen before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

        clock.tick(env.GAME_LOGIC_HZ) # Run at game logic speed

    env.close()