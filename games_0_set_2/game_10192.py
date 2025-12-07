import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Teleport through a collapsing grid, create clones to extend your reach, and survive as long as possible in hazardous zones."
    user_guide = "Use arrow keys to teleport. Press space to create a temporary clone which extends your teleport range."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 40
    GRID_W = SCREEN_WIDTH // GRID_SIZE
    GRID_H = SCREEN_HEIGHT // GRID_SIZE

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (25, 20, 50)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_CLONE = (150, 255, 200)
    COLOR_HAZARD = (255, 50, 50)
    COLOR_SAFE_ZONE = (50, 150, 255)
    COLOR_TEXT = (220, 220, 220)

    # Game Parameters
    PLAYER_RADIUS = 12
    MAX_EPISODE_STEPS = 5000
    INITIAL_TELEPORT_UNITS = 1
    MAX_TELEPORT_UNITS = 5
    CLONE_DURATION = 150 # steps
    CLONE_COOLDOWN = 300 # steps
    CLONE_TELEPORT_MULTIPLIER = 1.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.game_over_font = pygame.font.SysFont("Consolas", 48, bold=True)

        self.render_mode = render_mode
        self.player_pos = np.array([0.0, 0.0])
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.collapsing_areas = []
        self.safe_zones = []
        self.particles = []
        
        # Action state
        self.prev_space_held = False

        # Clone state
        self.clone_active = False
        self.clone_timer = 0
        self.clone_cooldown_timer = 0
        self.clone_pos = np.array([0.0, 0.0])

        # Difficulty scaling state
        self.teleport_units = self.INITIAL_TELEPORT_UNITS
        self.collapse_speed = 0.0
        self.collapse_spawn_interval = 0
        self.next_collapse_spawn = 0
        self.num_collapsing_areas_to_spawn = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        
        self.collapsing_areas.clear()
        self.safe_zones.clear()
        self.particles.clear()
        
        self.prev_space_held = False
        self.clone_active = False
        self.clone_timer = 0
        self.clone_cooldown_timer = 0

        # Reset difficulty
        self.teleport_units = self.INITIAL_TELEPORT_UNITS
        self.collapse_speed = 0.05
        self.collapse_spawn_interval = 200 # Initial longer period
        self.next_collapse_spawn = 100
        self.num_collapsing_areas_to_spawn = 5

        self._spawn_hazards_and_safe_zones()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # 1. Handle Input & Action Rewards
        action_reward = self._handle_input(action)
        reward += action_reward

        # 2. Update Game State
        self._update_game_state()
        
        # 3. Survival Reward
        reward += 0.1

        # 4. Check Termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        
        # 5. Update Score
        self.score += reward
        
        # 6. Advance Time
        self.steps += 1
        
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            terminated = True # Per Gym API, truncated episodes are also terminated
            reward += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, _ = action
        space_held = space_action == 1
        reward = 0.0
        
        # --- MOVEMENT (TELEPORT) ---
        if movement > 0:
            old_pos = self.player_pos.copy()
            
            teleport_range = self.teleport_units * self.GRID_SIZE
            if self.clone_active:
                teleport_range *= self.CLONE_TELEPORT_MULTIPLIER

            if movement == 1: self.player_pos[1] -= teleport_range # Up
            elif movement == 2: self.player_pos[1] += teleport_range # Down
            elif movement == 3: self.player_pos[0] -= teleport_range # Left
            elif movement == 4: self.player_pos[0] += teleport_range # Right
            
            self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
            self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

            self._create_particles(old_pos, 20, self.COLOR_PLAYER, -2) # Implosion
            self._create_particles(self.player_pos, 20, self.COLOR_PLAYER, 2) # Explosion

            player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
            for zone in self.safe_zones:
                if math.hypot(player_rect.centerx - zone['pos'][0], player_rect.centery - zone['pos'][1]) < zone['radius']:
                    reward += 1.0
                    break
        
        # --- CLONE ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed and self.clone_cooldown_timer <= 0:
            self.clone_active = True
            self.clone_timer = self.CLONE_DURATION
            self.clone_cooldown_timer = self.CLONE_COOLDOWN
            self.clone_pos = self.player_pos.copy()
            reward += 0.5
            self._create_particles(self.clone_pos, 30, self.COLOR_CLONE, 1.5)

        self.prev_space_held = space_held
        return reward

    def _update_game_state(self):
        self._update_difficulty()

        if self.clone_active:
            self.clone_timer -= 1
            if self.clone_timer <= 0:
                self.clone_active = False
                self._create_particles(self.clone_pos, 20, self.COLOR_CLONE, -1)
        if self.clone_cooldown_timer > 0:
            self.clone_cooldown_timer -= 1

        for area in self.collapsing_areas[:]:
            area['life'] -= self.collapse_speed
            if area['life'] <= 0:
                self.collapsing_areas.remove(area)

        if self.steps >= self.next_collapse_spawn:
            self._spawn_hazards_and_safe_zones()
            self.next_collapse_spawn = self.steps + self.collapse_spawn_interval

        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_difficulty(self):
        new_teleport_units = self.INITIAL_TELEPORT_UNITS + self.steps // 1000
        self.teleport_units = min(new_teleport_units, self.MAX_TELEPORT_UNITS)
        
        self.collapse_speed = 0.05 + (self.steps // 250) * 0.005

        spawn_interval_reduction = self.steps // 500
        self.collapse_spawn_interval = max(80, 200 - spawn_interval_reduction * 10)
        
        self.num_collapsing_areas_to_spawn = min(25, 5 + self.steps // 400)

    def _spawn_hazards_and_safe_zones(self):
        self.collapsing_areas.clear()
        self.safe_zones.clear()

        all_cells = [(i, j) for i in range(self.GRID_W) for j in range(self.GRID_H)]
        self.np_random.shuffle(all_cells)

        player_grid_x = int(self.player_pos[0] // self.GRID_SIZE)
        player_grid_y = int(self.player_pos[1] // self.GRID_SIZE)
        
        safe_spawn_cells = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                safe_spawn_cells.append((player_grid_x + i, player_grid_y + j))

        available_cells = [cell for cell in all_cells if cell not in safe_spawn_cells]
        
        num_safe_zones = max(2, 5 - self.steps // 1000)
        for i in range(num_safe_zones):
            if not available_cells: break
            grid_pos = available_cells.pop()
            center_x = grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2
            center_y = grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2
            self.safe_zones.append({
                'pos': (center_x, center_y),
                'radius': self.GRID_SIZE * 0.7
            })

        for i in range(self.num_collapsing_areas_to_spawn):
            if not available_cells: break
            grid_pos = available_cells.pop()
            rect = pygame.Rect(grid_pos[0] * self.GRID_SIZE, grid_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            self.collapsing_areas.append({'rect': rect, 'life': 1.0})

    def _check_termination(self):
        player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        player_rect.center = (self.player_pos[0], self.player_pos[1])

        for area in self.collapsing_areas:
            life_ratio = max(0, area['life'])
            collision_rect = area['rect'].inflate(
                area['rect'].width * (life_ratio - 1),
                area['rect'].height * (life_ratio - 1)
            )
            if player_rect.colliderect(collision_rect):
                self._create_particles(self.player_pos, 100, self.COLOR_HAZARD, 4)
                return True, 0

        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "teleport_range": self.teleport_units * self.GRID_SIZE,
            "clone_active": self.clone_active,
            "clone_cooldown": self.clone_cooldown_timer,
        }

    def _render_game(self):
        self._render_background()
        self._render_safe_zones()
        self._render_collapsing_areas()
        self._render_particles()
        if self.clone_active:
            self._render_clone()
        if not self.game_over:
            self._render_player()

    def _render_background(self):
        for i in range(self.GRID_W + 1):
            x = i * self.GRID_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for i in range(self.GRID_H + 1):
            y = i * self.GRID_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
            
    def _render_safe_zones(self):
        for zone in self.safe_zones:
            pulsate = 1 + 0.1 * math.sin(self.steps * 0.1)
            radius = int(zone['radius'] * pulsate)
            self._draw_glowing_circle(self.screen, self.COLOR_SAFE_ZONE, zone['pos'], radius, 4)

    def _render_collapsing_areas(self):
        for area in self.collapsing_areas:
            life_ratio = max(0, area['life'])
            
            glow_size = int(life_ratio * 10)
            if glow_size > 1:
                glow_rect = area['rect'].inflate(glow_size, glow_size)
                glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*self.COLOR_HAZARD, 50), glow_surf.get_rect(), border_radius=5)
                self.screen.blit(glow_surf, glow_rect.topleft)

            current_rect = area['rect'].inflate(
                area['rect'].width * (life_ratio - 1),
                area['rect'].height * (life_ratio - 1)
            )
            pygame.draw.rect(self.screen, self.COLOR_HAZARD, current_rect, 0, border_radius=3)
            
    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, pos, self.PLAYER_RADIUS, 8)
        
        if self.clone_cooldown_timer > 0:
            cooldown_ratio = self.clone_cooldown_timer / self.CLONE_COOLDOWN
            start_angle = -math.pi / 2
            end_angle = start_angle + (2 * math.pi * cooldown_ratio)
            
            arc_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS * 2.5, self.PLAYER_RADIUS * 2.5)
            arc_rect.center = pos
            pygame.draw.arc(self.screen, self.COLOR_TEXT, arc_rect, start_angle, end_angle, 2)

    def _render_clone(self):
        pos = (int(self.clone_pos[0]), int(self.clone_pos[1]))
        
        alpha_ratio = self.clone_timer / self.CLONE_DURATION
        alpha = int(150 * alpha_ratio)
        color = (*self.COLOR_CLONE, alpha)
        
        temp_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        center = (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        
        self._draw_glowing_circle(temp_surf, color, center, self.PLAYER_RADIUS, 6, use_alpha=True)
        self.screen.blit(temp_surf, (pos[0] - center[0], pos[1] - center[1]))

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            life_ratio = p['life'] / p['max_life']
            radius = int(p['radius'] * life_ratio)
            if radius > 0:
                color = (*p['color'], int(255 * life_ratio))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"TIME: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        if self.game_over:
            over_text = self.game_over_font.render("REALITY COLLAPSED", True, self.COLOR_HAZARD)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _draw_glowing_circle(self, surface, color, pos, radius, max_glow, use_alpha=False):
        pos = (int(pos[0]), int(pos[1]))
        for i in range(max_glow, 0, -1):
            alpha = int(100 * (1 - i / max_glow))
            if use_alpha:
                current_color = (*color[:3], int(color[3] * (1 - i / max_glow)**2))
            else:
                current_color = (*color, alpha)
            
            current_radius = int(radius + i * 1.5)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], current_radius, current_color)
        
        if use_alpha:
             pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
             pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        else:
             pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
             pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(20, 41)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'radius': self.np_random.integers(2, 6)
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for human play and is not part of the gym environment
    # It will not be checked by the test suite, but is useful for debugging
    render_mode = "human"
    if render_mode == "human":
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Quantum Collapse")
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        movement_action = 0 # None
        space_action = 0
        shift_action = 0

        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if render_mode == "human":
            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()