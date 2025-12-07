import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:41:11.619717
# Source Brief: brief_03199.md
# Brief Index: 3199
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player, a glowing orb, must evade quantum
    predators while collecting particles. The player can teleport and terraform
    the environment to create temporary cover.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Evade quantum predators as a glowing orb. Collect particles to upgrade your teleport ability and use terraforming to create temporary cover."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport. Press space to create a terraformed cover zone."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    TARGET_FPS = 30

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (25, 20, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 200)
    COLOR_PREDATOR = (255, 50, 50)
    COLOR_PREDATOR_GLOW = (200, 20, 20)
    COLOR_PARTICLE = (255, 255, 0)
    COLOR_TERRAFORM = (50, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_ENERGY_BAR = (0, 200, 255)
    COLOR_ENERGY_BAR_BG = (40, 40, 70)

    # Player
    PLAYER_RADIUS = 10
    INITIAL_TELEPORT_DISTANCE = 50
    TELEPORT_ENERGY_COST = 25
    TELEPORT_UPGRADE_DISTANCE_BONUS = 10

    # Predators
    NUM_PREDATORS = 3
    PREDATOR_SIZE = 12
    INITIAL_PREDATOR_SPEED = 1.5
    PREDATOR_SPEED_INCREASE = 0.2

    # Particles
    NUM_PARTICLES = 10

    # Terraform
    TERRAFORM_SIZE = 80
    TERRAFORM_DURATION = 300 # in steps
    TERRAFORM_COOLDOWN = 60 # in steps

    # Rewards
    REWARD_SURVIVAL = 0.01
    REWARD_PARTICLE_COLLECT = 10.0
    REWARD_UPGRADE = 25.0
    REWARD_TERMINAL_DEATH = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 64)

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.predators = []
        self.particles = []
        self.terraformed_areas = []
        self.effects = []
        self.teleport_energy = 0
        self.teleport_distance = 0
        self.predator_speed = 0
        self.particle_collect_count = 0
        self.next_upgrade_milestone = 0
        self.next_speed_milestone = 0
        self.terraform_cooldown_timer = 0
        
        # This will be called once to set up the initial state
        # self.reset()

        # Final check
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.teleport_energy = 100.0
        self.teleport_distance = self.INITIAL_TELEPORT_DISTANCE

        # Game progression state
        self.particle_collect_count = 0
        self.next_upgrade_milestone = 25
        self.next_speed_milestone = 50
        self.predator_speed = self.INITIAL_PREDATOR_SPEED
        self.terraform_cooldown_timer = 0

        # Entities
        self._spawn_predators()
        self._spawn_particles(self.NUM_PARTICLES)
        self.terraformed_areas = []
        self.effects = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = self.REWARD_SURVIVAL

        # --- UPDATE GAME STATE ---
        self._handle_input(action)
        self._update_predators()
        self._update_terraformed_areas()
        self._update_effects()

        # --- HANDLE INTERACTIONS & REWARDS ---
        reward += self._handle_collisions()
        reward += self._update_progression()

        self.score += reward

        # --- CHECK TERMINATION ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if self.game_over:
             reward = self.REWARD_TERMINAL_DEATH
             self.score += reward # Adjust score for the final penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action
        space_held = space_pressed == 1

        # Movement (Teleportation)
        if movement != 0 and self.teleport_energy >= self.TELEPORT_ENERGY_COST:
            # Sfx: Teleport_Whoosh
            self.teleport_energy -= self.TELEPORT_ENERGY_COST
            old_pos = self.player_pos.copy()
            
            direction_vectors = {
                1: np.array([0, -1]),  # Up
                2: np.array([0, 1]),   # Down
                3: np.array([-1, 0]),  # Left
                4: np.array([1, 0]),   # Right
            }
            direction = direction_vectors.get(movement, np.array([0, 0]))
            self.player_pos += direction * self.teleport_distance

            # Wrap around screen edges
            self.player_pos[0] %= self.SCREEN_WIDTH
            self.player_pos[1] %= self.SCREEN_HEIGHT
            
            self._create_effect('teleport', old_pos, duration=15)
            self._create_effect('teleport', self.player_pos.copy(), duration=15)

        # Regenerate energy
        self.teleport_energy = min(100.0, self.teleport_energy + 0.5)
        
        # Terraform
        self.terraform_cooldown_timer = max(0, self.terraform_cooldown_timer - 1)
        if space_held and self.terraform_cooldown_timer == 0:
            # Sfx: Terraform_Create
            self.terraform_cooldown_timer = self.TERRAFORM_COOLDOWN
            self.terraformed_areas.append({
                'rect': pygame.Rect(
                    self.player_pos[0] - self.TERRAFORM_SIZE / 2,
                    self.player_pos[1] - self.TERRAFORM_SIZE / 2,
                    self.TERRAFORM_SIZE,
                    self.TERRAFORM_SIZE
                ),
                'timer': self.TERRAFORM_DURATION
            })

    def _update_predators(self):
        for predator in self.predators:
            predator['pos'] += predator['vel'] * self.predator_speed
            
            # Bounce off walls
            if predator['pos'][0] < 0 or predator['pos'][0] > self.SCREEN_WIDTH:
                predator['vel'][0] *= -1
            if predator['pos'][1] < 0 or predator['pos'][1] > self.SCREEN_HEIGHT:
                predator['vel'][1] *= -1
            
            # Clamp position to prevent getting stuck
            predator['pos'][0] = np.clip(predator['pos'][0], 0, self.SCREEN_WIDTH)
            predator['pos'][1] = np.clip(predator['pos'][1], 0, self.SCREEN_HEIGHT)
    
    def _update_terraformed_areas(self):
        for area in self.terraformed_areas[:]:
            area['timer'] -= 1
            if area['timer'] <= 0:
                # Sfx: Terraform_Dissipate
                self.terraformed_areas.remove(area)
    
    def _update_progression(self):
        reward = 0
        # Teleport distance upgrade
        if self.particle_collect_count >= self.next_upgrade_milestone:
            # Sfx: Upgrade_Unlocked
            self.teleport_distance += self.TELEPORT_UPGRADE_DISTANCE_BONUS
            self.next_upgrade_milestone += 25
            reward += self.REWARD_UPGRADE
            self._create_effect('upgrade', self.player_pos.copy(), duration=45)
        
        # Predator speed increase
        if self.particle_collect_count >= self.next_speed_milestone:
            self.predator_speed += self.PREDATOR_SPEED_INCREASE
            self.next_speed_milestone += 50
        
        return reward

    def _is_player_in_cover(self):
        player_point = (int(self.player_pos[0]), int(self.player_pos[1]))
        for area in self.terraformed_areas:
            if area['rect'].collidepoint(player_point):
                return True
        return False

    def _handle_collisions(self):
        reward = 0
        
        # Player-Particle
        for particle_pos in self.particles[:]:
            dist = np.linalg.norm(self.player_pos - particle_pos)
            if dist < self.PLAYER_RADIUS + 3: # 3 is particle radius
                # Sfx: Particle_Collect
                self.particles.remove(particle_pos)
                self._spawn_particles(1)
                self.particle_collect_count += 1
                reward += self.REWARD_PARTICLE_COLLECT
                self._create_effect('collect', particle_pos, duration=20)

        # Player-Predator
        if not self._is_player_in_cover():
            for predator in self.predators:
                dist = np.linalg.norm(self.player_pos - predator['pos'])
                if dist < self.PLAYER_RADIUS + self.PREDATOR_SIZE:
                    # Sfx: Player_Death
                    self.game_over = True
                    self._create_effect('death', self.player_pos.copy(), duration=30)
                    break
        
        return reward

    def _spawn_predators(self):
        self.predators = []
        for _ in range(self.NUM_PREDATORS):
            pos = np.array([
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            ])
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)])
            self.predators.append({'pos': pos, 'vel': vel})

    def _spawn_particles(self, count):
        for _ in range(count):
            pos = [
                self.np_random.uniform(20, self.SCREEN_WIDTH - 20),
                self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
            ]
            self.particles.append(pos)
    
    def _create_effect(self, effect_type, pos, duration):
        self.effects.append({'type': effect_type, 'pos': pos, 'timer': duration, 'max_timer': duration})
        
    def _update_effects(self):
        for effect in self.effects[:]:
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                self.effects.remove(effect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "particles_collected": self.particle_collect_count,
            "teleport_energy": self.teleport_energy,
            "teleport_distance": self.teleport_distance,
        }

    # --- RENDERING METHODS ---
    def _render_game(self):
        self._render_background_grid()
        self._render_terraformed_areas()
        self._render_particles()
        self._render_predators()
        if not self.game_over:
            self._render_player()
        self._render_effects_draw()

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_terraformed_areas(self):
        for area in self.terraformed_areas:
            alpha = int(255 * (area['timer'] / self.TERRAFORM_DURATION))
            alpha = max(0, min(alpha, 255))
            
            # Create a temporary surface for transparency
            s = pygame.Surface((self.TERRAFORM_SIZE, self.TERRAFORM_SIZE), pygame.SRCALPHA)
            color = self.COLOR_TERRAFORM + (int(alpha * 0.3),) # Fill
            pygame.draw.rect(s, color, s.get_rect())
            color_border = self.COLOR_TERRAFORM + (alpha,) # Border
            pygame.draw.rect(s, color_border, s.get_rect(), 2)
            self.screen.blit(s, area['rect'].topleft)

    def _render_particles(self):
        for pos in self.particles:
            flicker = self.np_random.uniform(0.8, 1.2)
            radius = int(3 * flicker)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_PARTICLE)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_PARTICLE)

    def _render_predators(self):
        for predator in self.predators:
            pos = (int(predator['pos'][0]), int(predator['pos'][1]))
            size = self.PREDATOR_SIZE * (1 + 0.1 * math.sin(self.steps * 0.2)) # Oscillating size
            
            # Draw glow
            glow_radius = int(size * 1.8)
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.COLOR_PREDATOR_GLOW + (50,), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

            # Draw main triangle
            points = [
                (pos[0], pos[1] - size),
                (pos[0] - size * 0.866, pos[1] + size * 0.5),
                (pos[0] + size * 0.866, pos[1] + size * 0.5)
            ]
            angle = math.atan2(predator['vel'][1], predator['vel'][0]) + math.pi/2
            rotated_points = [
                (
                    pos[0] + (p[0] - pos[0]) * math.cos(angle) - (p[1] - pos[1]) * math.sin(angle),
                    pos[1] + (p[0] - pos[0]) * math.sin(angle) + (p[1] - pos[1]) * math.cos(angle)
                ) for p in points
            ]
            
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PREDATOR)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PREDATOR)
            
    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow effect
        glow_radius = int(self.PLAYER_RADIUS * 2.5)
        glow_center = (pos[0], pos[1])
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW + (30,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius))

        # Main body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        
        # Cover indicator
        if self._is_player_in_cover():
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 4, self.COLOR_TERRAFORM)

    def _render_effects_draw(self):
        for effect in self.effects:
            progress = 1 - (effect['timer'] / effect['max_timer'])
            pos = (int(effect['pos'][0]), int(effect['pos'][1]))
            
            if effect['type'] in ['teleport', 'death', 'upgrade']:
                num_particles = 20 if effect['type'] != 'upgrade' else 40
                color = self.COLOR_PLAYER if effect['type'] == 'teleport' else \
                        self.COLOR_PREDATOR if effect['type'] == 'death' else self.COLOR_PARTICLE
                
                for i in range(num_particles):
                    angle = (i / num_particles) * 2 * math.pi + (progress * 2)
                    dist = progress * 50
                    p_pos = (
                        int(pos[0] + math.cos(angle) * dist),
                        int(pos[1] + math.sin(angle) * dist)
                    )
                    alpha = int(255 * (1 - progress))
                    pygame.draw.circle(self.screen, color + (alpha,), p_pos, 2)
            
            elif effect['type'] == 'collect':
                for i in range(5):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    dist = progress * 20
                    p_pos = (
                        int(pos[0] + math.cos(angle) * dist),
                        int(pos[1] + math.sin(angle) * dist)
                    )
                    alpha = int(255 * (1-progress))
                    pygame.draw.circle(self.screen, self.COLOR_PARTICLE + (alpha,), p_pos, 1)

    def _render_ui(self):
        # Score and Particles
        score_text = f"SCORE: {int(self.score)}"
        particle_text = f"PARTICLES: {self.particle_collect_count}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        particle_surf = self.font_ui.render(particle_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(particle_surf, (10, 35))

        # Teleport Energy Bar
        bar_width = 150
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        energy_ratio = self.teleport_energy / 100.0
        
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (bar_x, bar_y, bar_width * energy_ratio, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        
    def _render_game_over(self):
        text_surf = self.font_game_over.render("GAME OVER", True, self.COLOR_PREDATOR)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        # Add a semi-transparent background for readability
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to initialize attributes
        self.reset()
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset return values
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
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    # env.validate_implementation() # Optional validation
    obs, info = env.reset()

    # --- Pygame setup for human play ---
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Evade")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    print("--- RESET ---")
                    obs, info = env.reset()
                    total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for 'r' to reset
            pass
        
        clock.tick(GameEnv.TARGET_FPS)

    env.close()