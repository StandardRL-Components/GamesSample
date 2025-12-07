import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:11:57.070185
# Source Brief: brief_00921.md
# Brief Index: 921
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment simulating protein synthesis within a cell.
    The agent controls a selector to move proteins, aligning them to a target sequence.
    Successful synthesis increases score and health, while mismatches are damaging.
    The goal is to complete a set number of synthesis chains before the cell dies.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide proteins to the synthesis zone to match the target sequence. "
        "Correct matches build the chain, but mismatches damage the cell."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected protein. "
        "Press Shift to cycle selection and Space to attempt synthesis in the designated zone."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 20

        # Colors
        self.COLOR_BG = (15, 20, 45)
        self.COLOR_MEMBRANE = (50, 60, 100)
        self.COLOR_MEMBRANE_HIGHLIGHT = (80, 90, 130)
        self.PROTEIN_COLORS = {
            "red": (255, 80, 80),
            "green": (80, 255, 80),
            "blue": (80, 80, 255),
            "yellow": (255, 255, 80),
        }
        self.COLOR_NAMES = list(self.PROTEIN_COLORS.keys())
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR = (100, 220, 100)
        self.COLOR_HEALTH_BAR_BG = (220, 100, 100)
        self.COLOR_SELECTED_GLOW = (255, 255, 255)
        self.COLOR_SYNTHESIS_ZONE = (70, 80, 120)
        self.COLOR_SYNTHESIS_ZONE_ACTIVE = (120, 140, 200)

        # Game parameters
        self.PLAYER_SPEED = 5.0
        self.PROTEIN_SIZE = 16
        self.CELL_RADIUS = 180
        self.CELL_CENTER = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2 + 20)
        self.MAX_PROTEINS = 12
        self.SYNTHESIS_COOLDOWN_FRAMES = 10 # Prevent mashing synthesis
        self.TARGET_SEQUENCE_LENGTH = 5

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
        try:
            self.font_small = pygame.font.SysFont("Arial", 16)
            self.font_large = pygame.font.SysFont("Arial", 24)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.syntheses_count = 0
        self.cell_health = 0.0
        self.proteins = []
        self.particles = []
        self.selected_protein_idx = 0
        self.last_shift_state = False
        self.synthesis_cooldown = 0
        self.protein_base_speed = 0.0
        self.target_sequence = []
        self.current_sequence_idx = 0
        self.synthesis_zone_rect = None
        self.game_over = False

        # Initialize state variables for the first time
        # self.reset() is called by the wrapper, no need to call it here.
        
        # --- Final Validation ---
        # self.validate_implementation() # This should not be in the final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.syntheses_count = 0
        self.cell_health = 100.0
        self.game_over = False
        
        self.proteins.clear()
        self.particles.clear()
        
        self.selected_protein_idx = 0
        self.last_shift_state = False
        self.synthesis_cooldown = 0
        self.protein_base_speed = 0.5
        self.current_sequence_idx = 0

        self.synthesis_zone_rect = pygame.Rect(
            self.CELL_CENTER.x - 50, self.HEIGHT - 50, 100, 40
        )

        self._generate_target_sequence()
        self._spawn_initial_proteins()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # Update cooldowns
        if self.synthesis_cooldown > 0:
            self.synthesis_cooldown -= 1

        # 1. Handle player input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)
        
        # 2. Update game logic
        self._update_proteins()
        self._update_particles()
        
        # 3. Calculate continuous reward for positioning
        reward += self._calculate_positioning_reward()

        # 4. Check for termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.syntheses_count >= self.WIN_SCORE:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        
        # Action: Cycle selection (Shift key)
        if shift_held and not self.last_shift_state and len(self.proteins) > 0:
            # On rising edge of shift press
            self.selected_protein_idx = (self.selected_protein_idx + 1) % len(self.proteins)
            # SFX: Play selection sound
        self.last_shift_state = shift_held

        # Action: Move selected protein (Arrow keys)
        if len(self.proteins) > 0:
            selected_p = self.proteins[self.selected_protein_idx]
            move_vec = pygame.Vector2(0, 0)
            if movement == 1: move_vec.y -= 1 # Up
            elif movement == 2: move_vec.y += 1 # Down
            elif movement == 3: move_vec.x -= 1 # Left
            elif movement == 4: move_vec.x += 1 # Right
            
            if move_vec.length() > 0:
                selected_p["pos"] += move_vec.normalize() * self.PLAYER_SPEED

        # Action: Attempt synthesis (Space key)
        if space_held and self.synthesis_cooldown == 0:
            reward += self._attempt_synthesis()
            self.synthesis_cooldown = self.SYNTHESIS_COOLDOWN_FRAMES

        return reward

    def _attempt_synthesis(self):
        if not self.proteins:
            return 0
            
        selected_p = self.proteins[self.selected_protein_idx]
        protein_rect = pygame.Rect(selected_p["pos"].x - self.PROTEIN_SIZE / 2, 
                                   selected_p["pos"].y - self.PROTEIN_SIZE / 2, 
                                   self.PROTEIN_SIZE, self.PROTEIN_SIZE)

        if self.synthesis_zone_rect.colliderect(protein_rect):
            target_color_name = self.target_sequence[self.current_sequence_idx]
            
            if selected_p["color_name"] == target_color_name:
                # --- SUCCESSFUL SYNTHESIS ---
                self.score += 10
                self.syntheses_count += 1
                self.cell_health = min(100, self.cell_health + 5)
                self._create_particles(selected_p["pos"], self.PROTEIN_COLORS[target_color_name], 30, is_success=True)
                # SFX: Play success chime
                
                # Remove synthesized protein and select next
                self.proteins.pop(self.selected_protein_idx)
                if len(self.proteins) > 0:
                    self.selected_protein_idx %= len(self.proteins)
                
                # Advance sequence
                self.current_sequence_idx += 1
                if self.current_sequence_idx >= len(self.target_sequence):
                    self._generate_target_sequence()

                # Increase difficulty every 5 syntheses
                if self.syntheses_count > 0 and self.syntheses_count % 5 == 0:
                    self.protein_base_speed = min(2.0, self.protein_base_speed + 0.15)

                return 10.0
            else:
                # --- FAILED SYNTHESIS ---
                self.cell_health = max(0, self.cell_health - 15)
                self._create_particles(selected_p["pos"], (100, 100, 100), 20, is_success=False)
                # SFX: Play failure buzz
                return -5.0
        return 0.0

    def _calculate_positioning_reward(self):
        if not self.proteins:
            return 0

        selected_p = self.proteins[self.selected_protein_idx]
        protein_rect = pygame.Rect(selected_p["pos"].x - self.PROTEIN_SIZE / 2, 
                                   selected_p["pos"].y - self.PROTEIN_SIZE / 2, 
                                   self.PROTEIN_SIZE, self.PROTEIN_SIZE)
        
        if self.synthesis_zone_rect.colliderect(protein_rect):
            target_color_name = self.target_sequence[self.current_sequence_idx]
            if selected_p["color_name"] == target_color_name:
                return 0.1 # Small reward for correct positioning
            else:
                return -0.1 # Small penalty for incorrect positioning
        return 0.0

    def _update_proteins(self):
        # Spawn new proteins if needed
        while len(self.proteins) < self.MAX_PROTEINS:
            self._spawn_protein()

        for i, p in enumerate(self.proteins):
            # Update drifting for non-selected proteins
            if i != self.selected_protein_idx:
                p["pos"] += p["vel"]
            
            # Boundary collision with cell wall
            dist_from_center = p["pos"].distance_to(self.CELL_CENTER)
            if dist_from_center > self.CELL_RADIUS - self.PROTEIN_SIZE / 2:
                # Reflect velocity and move back inside
                normal = (self.CELL_CENTER - p["pos"]).normalize()
                p["vel"] = p["vel"].reflect(normal)
                p["pos"] = self.CELL_CENTER - normal * (self.CELL_RADIUS - self.PROTEIN_SIZE / 2)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] = max(0, p["radius"] * 0.95)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cell_health": self.cell_health,
            "syntheses_count": self.syntheses_count,
        }

    def _check_termination(self):
        return (
            self.cell_health <= 0
            or self.syntheses_count >= self.WIN_SCORE
        )
        
    def _render_game(self):
        # Cell membrane
        pygame.gfxdraw.aacircle(self.screen, int(self.CELL_CENTER.x), int(self.CELL_CENTER.y), self.CELL_RADIUS, self.COLOR_MEMBRANE)
        pygame.gfxdraw.filled_circle(self.screen, int(self.CELL_CENTER.x), int(self.CELL_CENTER.y), self.CELL_RADIUS, self.COLOR_MEMBRANE)
        pygame.gfxdraw.aacircle(self.screen, int(self.CELL_CENTER.x), int(self.CELL_CENTER.y), self.CELL_RADIUS-2, self.COLOR_MEMBRANE_HIGHLIGHT)

        # Synthesis zone
        zone_color = self.COLOR_SYNTHESIS_ZONE_ACTIVE if self.synthesis_cooldown > 0 else self.COLOR_SYNTHESIS_ZONE
        pygame.draw.rect(self.screen, zone_color, self.synthesis_zone_rect, border_radius=5)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # Proteins
        for i, p in enumerate(self.proteins):
            pos = (int(p["pos"].x), int(p["pos"].y))
            color = self.PROTEIN_COLORS[p["color_name"]]
            
            # Glow for selected protein
            if i == self.selected_protein_idx and not self.game_over:
                for j in range(5, 0, -1):
                    glow_alpha = 100 - j * 15
                    glow_color = (*self.COLOR_SELECTED_GLOW, glow_alpha)
                    s = pygame.Surface((self.PROTEIN_SIZE + j*4, self.PROTEIN_SIZE + j*4), pygame.SRCALPHA)
                    pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=6)
                    self.screen.blit(s, (pos[0] - self.PROTEIN_SIZE/2 - j*2, pos[1] - self.PROTEIN_SIZE/2 - j*2))

            # Protein body
            rect = pygame.Rect(0, 0, self.PROTEIN_SIZE, self.PROTEIN_SIZE)
            rect.center = pos
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, width=2, border_radius=3)

    def _render_ui(self):
        # Health bar
        health_bar_width = 200
        health_ratio = max(0, self.cell_health / 100.0)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, 20), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, health_bar_width * health_ratio, 20), border_radius=5)
        health_text = self.font_small.render(f"Cell Health", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score and Syntheses
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 5))
        syntheses_text = self.font_small.render(f"Chains: {self.syntheses_count} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(syntheses_text, (self.WIDTH - syntheses_text.get_width() - 10, 35))

        # Target Sequence
        seq_label = self.font_small.render("Target Sequence:", True, self.COLOR_UI_TEXT)
        self.screen.blit(seq_label, (10, self.HEIGHT - 30))
        for i, color_name in enumerate(self.target_sequence):
            color = self.PROTEIN_COLORS[color_name]
            rect = pygame.Rect(130 + i * 25, self.HEIGHT - 32, 20, 20)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            if i == self.current_sequence_idx:
                pygame.draw.rect(self.screen, self.COLOR_SELECTED_GLOW, rect, width=2, border_radius=3)

    def _spawn_protein(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.np_random.uniform(0, self.CELL_RADIUS * 0.8)
        pos = self.CELL_CENTER + pygame.Vector2(math.cos(angle), math.sin(angle)) * dist
        
        vel_angle = self.np_random.uniform(0, 2 * math.pi)
        vel = pygame.Vector2(math.cos(vel_angle), math.sin(vel_angle)) * self.protein_base_speed
        
        # Introduce new colors based on progress
        num_colors = 3 if self.syntheses_count < 5 else 4
        color_name = self.np_random.choice(self.COLOR_NAMES[:num_colors])
        
        self.proteins.append({"pos": pos, "vel": vel, "color_name": color_name})

    def _spawn_initial_proteins(self):
        for _ in range(self.MAX_PROTEINS):
            self._spawn_protein()

    def _generate_target_sequence(self):
        self.target_sequence.clear()
        self.current_sequence_idx = 0
        num_colors = 3 if self.syntheses_count < 5 else 4
        for _ in range(self.TARGET_SEQUENCE_LENGTH):
            self.target_sequence.append(self.np_random.choice(self.COLOR_NAMES[:num_colors]))

    def _create_particles(self, pos, color, count, is_success):
        for _ in range(count):
            if is_success:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                lifespan = self.np_random.integers(15, 30)
                radius = self.np_random.uniform(2, 6)
            else: # Failure particles are less energetic
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, 2)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                lifespan = self.np_random.integers(10, 20)
                radius = self.np_random.uniform(1, 4)

            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "radius": radius,
                "color": color
            })

    def close(self):
        pygame.quit()

    def render(self):
        # This method is not used in the standard gym loop with render_mode="rgb_array"
        # but can be useful for human-playable versions.
        return self._get_observation()


if __name__ == "__main__":
    # Example of how to run the environment for human play
    # This part of the script will not run in a headless environment
    # so we need to unset the SDL_VIDEODRIVER variable.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display window
    pygame.display.set_caption("Protein Synthesis")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # None
        space_held = 0 # Released
        shift_held = 0 # Released

        # This is a simplified rising-edge detector for shift
        current_shift_state = pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]
        if current_shift_state and not env.last_shift_state:
            shift_held = 1
        env.last_shift_state = current_shift_state
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Game will auto-pause, press R to reset

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()