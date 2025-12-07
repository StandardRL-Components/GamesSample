import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls two repelling magnets
    to collect metal scraps in a shrinking, oscillating arena.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two repelling magnets to collect metal scraps in a shrinking, oscillating arena before it collapses."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to apply force to both magnets. Collect scraps to score points."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = pygame.math.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_ARENA = (200, 220, 255)
    COLOR_MAGNET_1 = (255, 50, 80)
    COLOR_MAGNET_1_GLOW = (255, 50, 80, 40)
    COLOR_MAGNET_2 = (50, 150, 255)
    COLOR_MAGNET_2_GLOW = (50, 150, 255, 40)
    COLOR_SCRAP = (180, 180, 190)
    COLOR_PARTICLE = (255, 200, 80)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_TEXT_WARN = (255, 100, 100)

    # Game Parameters
    MAX_STEPS = 1500
    WIN_SCORE = 50
    NUM_SCRAPS = 30
    INITIAL_ARENA_RADIUS = 180
    MIN_ARENA_RADIUS_FACTOR = 0.15

    # Physics Parameters
    MAGNET_RADIUS = 12
    SCRAP_SIZE = 6
    MAGNET_MOVE_FORCE = 1.8
    MAGNET_REPEL_STRENGTH = 800.0
    SCRAP_ATTRACT_STRENGTH = 400.0
    MIN_FORCE_DIST_SQ = 50.0  # To prevent infinite forces
    DAMPING = 0.96

    # Arena Shrinking & Oscillation
    ARENA_SHRINK_RATE_BASE = 0.0005  # 0.05% of initial radius per step
    ARENA_SHRINK_RATE_PER_SCRAP = 0.00002 # 0.002% increase per scrap
    ARENA_OSC_AMPLITUDE = 6
    ARENA_OSC_FREQUENCY = 0.05

    # Rewards
    REWARD_SCRAP = 0.1
    REWARD_CHAIN_REACTION = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0
    CHAIN_REACTION_THRESHOLD = 5
    CHAIN_REACTION_DURATION = 90 # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_ui_large = pygame.font.Font(None, 48)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.magnet1_pos = pygame.math.Vector2(0, 0)
        self.magnet1_vel = pygame.math.Vector2(0, 0)
        self.magnet2_pos = pygame.math.Vector2(0, 0)
        self.magnet2_vel = pygame.math.Vector2(0, 0)
        self.scraps = []
        self.particles = []
        self.current_arena_radius = self.INITIAL_ARENA_RADIUS
        self.chain_reaction_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Reset magnets
        self.magnet1_pos = self.CENTER + pygame.math.Vector2(-50, 0)
        self.magnet1_vel = pygame.math.Vector2(0, 0)
        self.magnet2_pos = self.CENTER + pygame.math.Vector2(50, 0)
        self.magnet2_vel = pygame.math.Vector2(0, 0)
        
        # Reset arena
        self.current_arena_radius = self.INITIAL_ARENA_RADIUS
        self.chain_reaction_timer = 0

        # Reset scraps and particles
        self.scraps = []
        self.particles = []
        for _ in range(self.NUM_SCRAPS):
            self._spawn_scrap()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Actions ---
        movement, _, _ = action # space and shift are unused
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        move_force = move_vec * self.MAGNET_MOVE_FORCE
        if self.chain_reaction_timer > 0:
            move_force *= 2.0 # Speed boost during chain reaction

        # --- 2. Physics Simulation ---
        # Initialize forces
        force1 = pygame.math.Vector2(move_force)
        force2 = pygame.math.Vector2(move_force)

        # Magnet-Magnet repulsion
        vec_1_to_2 = self.magnet2_pos - self.magnet1_pos
        dist_sq_12 = vec_1_to_2.length_squared()
        if dist_sq_12 > 0:
            repel_force_mag = self.MAGNET_REPEL_STRENGTH / max(dist_sq_12, self.MIN_FORCE_DIST_SQ)
            repel_force_vec = vec_1_to_2.normalize() * repel_force_mag
            force1 -= repel_force_vec
            force2 += repel_force_vec

        # Magnet-Scrap attraction
        for scrap in self.scraps:
            # From magnet 1
            vec_m1_to_s = scrap['pos'] - self.magnet1_pos
            dist_sq_m1s = vec_m1_to_s.length_squared()
            if dist_sq_m1s > 0:
                attr_force_mag1 = self.SCRAP_ATTRACT_STRENGTH / max(dist_sq_m1s, self.MIN_FORCE_DIST_SQ)
                scrap['vel'] -= vec_m1_to_s.normalize() * attr_force_mag1
            
            # From magnet 2
            vec_m2_to_s = scrap['pos'] - self.magnet2_pos
            dist_sq_m2s = vec_m2_to_s.length_squared()
            if dist_sq_m2s > 0:
                attr_force_mag2 = self.SCRAP_ATTRACT_STRENGTH / max(dist_sq_m2s, self.MIN_FORCE_DIST_SQ)
                scrap['vel'] -= vec_m2_to_s.normalize() * attr_force_mag2
        
        # Update magnet velocities and positions
        self.magnet1_vel += force1
        self.magnet1_vel *= self.DAMPING
        self.magnet1_pos += self.magnet1_vel

        self.magnet2_vel += force2
        self.magnet2_vel *= self.DAMPING
        self.magnet2_pos += self.magnet2_vel

        # Update scrap velocities and positions
        for scrap in self.scraps:
            scrap['vel'] *= self.DAMPING
            scrap['pos'] += scrap['vel']

        # --- 3. Arena and Collision Logic ---
        # Update arena radius (linear shrink + oscillation for collision)
        shrink_rate = self.ARENA_SHRINK_RATE_BASE + self.score * self.ARENA_SHRINK_RATE_PER_SCRAP
        self.current_arena_radius -= shrink_rate * self.INITIAL_ARENA_RADIUS
        self.current_arena_radius = max(self.current_arena_radius, self.INITIAL_ARENA_RADIUS * self.MIN_ARENA_RADIUS_FACTOR)
        
        collision_radius = self.current_arena_radius + self.ARENA_OSC_AMPLITUDE * math.sin(self.steps * self.ARENA_OSC_FREQUENCY)

        # Boundary checks
        self._constrain_to_arena(self.magnet1_pos, self.magnet1_vel, self.MAGNET_RADIUS, collision_radius)
        self._constrain_to_arena(self.magnet2_pos, self.magnet2_vel, self.MAGNET_RADIUS, collision_radius)
        for scrap in self.scraps:
            self._constrain_to_arena(scrap['pos'], scrap['vel'], self.SCRAP_SIZE / 2, collision_radius)

        # --- 4. Game Rules ---
        # Scrap collection
        scraps_collected_this_step = 0
        scraps_to_remove = []
        for scrap in self.scraps:
            collected = False
            if (self.magnet1_pos - scrap['pos']).length() < self.MAGNET_RADIUS + self.SCRAP_SIZE / 2:
                collected = True
            if not collected and (self.magnet2_pos - scrap['pos']).length() < self.MAGNET_RADIUS + self.SCRAP_SIZE / 2:
                collected = True
            
            if collected:
                scraps_to_remove.append(scrap)
                scraps_collected_this_step += 1
                self.score += 1
                reward += self.REWARD_SCRAP
                self._spawn_particles(scrap['pos'], 10)

        if scraps_to_remove:
            self.scraps = [s for s in self.scraps if s not in scraps_to_remove]
            for _ in scraps_to_remove:
                self._spawn_scrap()

        # Chain reaction
        if scraps_collected_this_step >= self.CHAIN_REACTION_THRESHOLD:
            self.chain_reaction_timer = self.CHAIN_REACTION_DURATION
            reward += self.REWARD_CHAIN_REACTION
        
        if self.chain_reaction_timer > 0:
            self.chain_reaction_timer -= 1

        # Particle update
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 0.04
            p['size'] -= 0.1

        # --- 5. Termination Check ---
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += self.REWARD_WIN
        elif self.current_arena_radius <= self.INITIAL_ARENA_RADIUS * self.MIN_ARENA_RADIUS_FACTOR:
            terminated = True
            reward += self.REWARD_LOSE
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        if terminated or truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_scrap(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        # Spawn further out to keep the center clear
        dist = self.np_random.uniform(self.INITIAL_ARENA_RADIUS * 0.4, self.INITIAL_ARENA_RADIUS * 0.95)
        pos = self.CENTER + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * dist
        self.scraps.append({'pos': pos, 'vel': pygame.math.Vector2(0, 0)})

    def _spawn_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 3.0)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': 1.0,
                'size': self.np_random.uniform(2, 5)
            })

    def _constrain_to_arena(self, pos, vel, radius, arena_radius):
        vec_from_center = pos - self.CENTER
        dist = vec_from_center.length()
        if dist > arena_radius - radius:
            # Push back inside
            pos.x = self.CENTER.x + (arena_radius - radius) * vec_from_center.x / dist
            pos.y = self.CENTER.y + (arena_radius - radius) * vec_from_center.y / dist
            # Reflect velocity
            normal = vec_from_center.normalize()
            vel.reflect_ip(normal)
            vel *= 0.8 # Energy loss on collision

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render arena
        display_radius = self.current_arena_radius + self.ARENA_OSC_AMPLITUDE * math.sin(self.steps * self.ARENA_OSC_FREQUENCY)
        pygame.gfxdraw.aacircle(self.screen, int(self.CENTER.x), int(self.CENTER.y), int(display_radius), self.COLOR_ARENA)
        
        # Render scraps
        for scrap in self.scraps:
            pygame.draw.rect(self.screen, self.COLOR_SCRAP, (
                int(scrap['pos'].x - self.SCRAP_SIZE / 2), 
                int(scrap['pos'].y - self.SCRAP_SIZE / 2),
                self.SCRAP_SIZE, self.SCRAP_SIZE
            ))
        
        # Render magnets
        self._render_magnet(self.magnet1_pos, self.COLOR_MAGNET_1, self.COLOR_MAGNET_1_GLOW)
        self._render_magnet(self.magnet2_pos, self.COLOR_MAGNET_2, self.COLOR_MAGNET_2_GLOW)

        # Render particles
        for p in self.particles:
            if p['life'] > 0 and p['size'] > 0:
                alpha = max(0, min(255, int(p['life'] * 255)))
                color = (*self.COLOR_PARTICLE, alpha)
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (int(p['size']), int(p['size'])), int(p['size']))
                self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))
        
        # Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_magnet(self, pos, color, glow_color):
        # Glow effect
        glow_radius = self.MAGNET_RADIUS * (2.5 if self.chain_reaction_timer > 0 else 2.0)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, int(glow_radius), int(glow_radius), int(glow_radius), glow_color)
        self.screen.blit(temp_surf, (int(pos.x - glow_radius), int(pos.y - glow_radius)))
        
        # Main magnet body
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.MAGNET_RADIUS, color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.MAGNET_RADIUS, color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Scraps: {self.score} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Arena Size
        arena_percent = (self.current_arena_radius / self.INITIAL_ARENA_RADIUS) * 100
        arena_color = self.COLOR_UI_TEXT if arena_percent > 30 else self.COLOR_UI_TEXT_WARN
        arena_text = self.font_ui.render(f"Arena: {arena_percent:.0f}%", True, arena_color)
        self.screen.blit(arena_text, (self.SCREEN_WIDTH - arena_text.get_width() - 10, 10))

        # Chain reaction timer
        if self.chain_reaction_timer > 0:
            chain_text = self.font_ui_large.render("CHAIN REACTION!", True, self.COLOR_PARTICLE)
            text_rect = chain_text.get_rect(center=(self.CENTER.x, 30))
            self.screen.blit(chain_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "arena_radius_percent": (self.current_arena_radius / self.INITIAL_ARENA_RADIUS) * 100,
            "chain_reaction_active": self.chain_reaction_timer > 0,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


# Example usage:
if __name__ == '__main__':
    # This block is for human play and visualization, and is not used by the tests.
    # It requires a display to be available.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Magnet Mayhem")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(60) # Limit to 60 FPS for smooth play

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()