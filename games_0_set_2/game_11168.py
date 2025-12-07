import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:33:27.436119
# Source Brief: brief_01168.md
# Brief Index: 1168
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper function for linear interpolation
def lerp(a, b, t):
    return a + (b - a) * t

# Helper function to draw text
def draw_text(surface, text, pos, font, color, center=False):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center:
        text_rect.center = pos
    else:
        text_rect.topleft = pos
    surface.blit(text_surface, text_rect)

# Helper function for glow effect
def draw_glow_circle(surface, pos, radius, color, glow_strength=5):
    for i in range(glow_strength, 0, -1):
        alpha = int(100 * (1 - i / glow_strength))
        glow_color = (*color, alpha)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius + i * 2), glow_color)
    pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius), color)
    pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius), color)

class HopliteSquad:
    def __init__(self, squad_id, pos, is_player=True, size=15):
        self.id = squad_id
        self.pos = np.array(pos, dtype=float)
        self.visual_pos = np.array(pos, dtype=float)
        self.is_player = is_player
        self.size = size
        self.patrol_index = 0
        self.patrol_path = []
        self.trail = []

    def move_to(self, new_pos):
        self.pos[:] = new_pos

    def update_visuals(self, lerp_factor):
        self.visual_pos = lerp(self.visual_pos, self.pos, lerp_factor)
        self.trail.append(self.visual_pos.copy())
        if len(self.trail) > 20:
            self.trail.pop(0)

class Citadel:
    def __init__(self, citadel_id, pos, size=25):
        self.id = citadel_id
        self.pos = np.array(pos, dtype=float)
        self.size = size
        self.owner = "enemy"  # Can be 'enemy' or 'player'

class Portal:
    def __init__(self, portal_id, pos, size=20):
        self.id = portal_id
        self.pos = np.array(pos, dtype=float)
        self.size = size

class Particle:
    def __init__(self, pos, vel, lifespan, color, size):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.size = size

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        return self.lifespan > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A real-time strategy game where you command hoplite squads through portals to capture enemy citadels."
    )
    user_guide = (
        "Use arrow keys to cycle between squads and portals. Press space to select a target and confirm the move. Press shift to cancel."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 80
    MAX_STEPS = 1000
    BEAT_DURATION = 30  # 1 beat per second at 30fps

    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_CITADEL_ENEMY = (200, 150, 0)
    COLOR_CITADEL_PLAYER = (0, 200, 200)
    COLOR_PORTAL = (150, 0, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PROMPT = (255, 255, 100)

    REWARD_WIN = 100
    REWARD_LOSS = -100
    REWARD_CAPTURE_CITADEL = 10
    REWARD_LOSE_SQUAD = -5
    REWARD_MOVE_CLOSER = 0.1
    REWARD_MOVE_FARTHER = -0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 36, bold=True)

        self.player_squads = []
        self.enemy_squads = []
        self.citadels = []
        self.portals = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.resources = 10
        self.captured_citadels_count = 0
        self.enemy_speed_multiplier = 1.0

        self.player_squads.clear()
        self.enemy_squads.clear()
        self.citadels.clear()
        self.portals.clear()
        self.particles.clear()

        self._setup_level()

        self.game_mode = 'SELECT_SQUAD'
        self.selected_squad_idx = 0
        self.selected_portal_idx = 0
        self.queued_move = None
        self.last_space_held = False
        self.last_shift_held = False
        self.beat_timer = 0

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        # Player Squads
        self.player_squads.append(HopliteSquad(0, (self.GRID_SIZE * 1.5, self.HEIGHT / 2)))
        self.player_squads.append(HopliteSquad(1, (self.GRID_SIZE * 1.5, self.HEIGHT / 2 + self.GRID_SIZE)))

        # Citadels
        self.citadels.append(Citadel(0, (self.WIDTH - self.GRID_SIZE * 1.5, self.GRID_SIZE * 1.5)))
        self.citadels.append(Citadel(1, (self.WIDTH - self.GRID_SIZE * 1.5, self.HEIGHT - self.GRID_SIZE * 1.5)))
        self.citadels.append(Citadel(2, (self.WIDTH / 2, self.GRID_SIZE * 1.0)))

        # Portals
        self.portals.append(Portal(0, (self.GRID_SIZE * 3, self.GRID_SIZE * 1)))
        self.portals.append(Portal(1, (self.GRID_SIZE * 3, self.HEIGHT - self.GRID_SIZE * 1)))
        self.portals.append(Portal(2, (self.WIDTH - self.GRID_SIZE * 3, self.GRID_SIZE * 2)))
        self.portals.append(Portal(3, (self.WIDTH - self.GRID_SIZE * 3, self.HEIGHT - self.GRID_SIZE * 2)))

        # Enemy Squads
        self._add_enemy_squads(2)

    def _add_enemy_squads(self, count):
        for i in range(count):
            start_y = self.np_random.uniform(self.GRID_SIZE, self.HEIGHT - self.GRID_SIZE)
            squad = HopliteSquad(len(self.enemy_squads), (self.WIDTH - self.GRID_SIZE, start_y), is_player=False)
            squad.patrol_path = [np.array([self.WIDTH - self.GRID_SIZE, start_y]), np.array([self.WIDTH / 2 + self.GRID_SIZE, start_y])]
            self.enemy_squads.append(squad)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.beat_timer += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- Handle Input ---
        if self.game_mode == 'SELECT_SQUAD':
            if movement in [1, 2] and len(self.player_squads) > 1: # Up/Down
                self.selected_squad_idx = (self.selected_squad_idx + (1 if movement == 2 else -1)) % len(self.player_squads)
            elif space_pressed and self.player_squads:
                self.game_mode = 'SELECT_PORTAL'
                self.selected_portal_idx = 0
                # Sound: UI_Select

        elif self.game_mode == 'SELECT_PORTAL':
            if movement in [1, 2, 3, 4] and len(self.portals) > 1: # Any direction
                self.selected_portal_idx = (self.selected_portal_idx + (1 if movement in [2,4] else -1)) % len(self.portals)
            elif shift_pressed:
                self.game_mode = 'SELECT_SQUAD'
                # Sound: UI_Cancel
            elif space_pressed and self.player_squads and self.portals:
                squad = self.player_squads[self.selected_squad_idx]
                portal = self.portals[self.selected_portal_idx]
                
                # Calculate distance-based reward before move
                old_dist = min(np.linalg.norm(squad.pos - c.pos) for c in self.citadels if c.owner == 'enemy') if any(c.owner == 'enemy' for c in self.citadels) else 0
                new_dist = min(np.linalg.norm(portal.pos - c.pos) for c in self.citadels if c.owner == 'enemy') if any(c.owner == 'enemy' for c in self.citadels) else 0
                if new_dist < old_dist:
                    reward += self.REWARD_MOVE_CLOSER
                else:
                    reward += self.REWARD_MOVE_FARTHER

                self.queued_move = (self.selected_squad_idx, self.selected_portal_idx)
                self.game_mode = 'SELECT_SQUAD'
                # Sound: UI_Confirm

        # --- Handle Beat ---
        if self.beat_timer >= self.BEAT_DURATION:
            self.beat_timer = 0
            # Sound: Beat_Tick
            
            # Execute Player Move
            if self.queued_move is not None and self.player_squads:
                squad_idx, portal_idx = self.queued_move
                if squad_idx < len(self.player_squads):
                    squad = self.player_squads[squad_idx]
                    portal = self.portals[portal_idx]
                    squad.move_to(portal.pos)
                    # Sound: Player_Warp
                    self._create_particle_burst(squad.visual_pos, 20, self.COLOR_PLAYER)
                    self._create_particle_burst(portal.pos, 20, self.COLOR_PORTAL)
                self.queued_move = None

            # Move Enemies
            for enemy in self.enemy_squads:
                if not enemy.patrol_path: continue
                target = enemy.patrol_path[enemy.patrol_index]
                if np.linalg.norm(enemy.pos - target) < 5:
                    enemy.patrol_index = (enemy.patrol_index + 1) % len(enemy.patrol_path)
                
                direction = enemy.patrol_path[enemy.patrol_index] - enemy.pos
                dist = np.linalg.norm(direction)
                if dist > 0:
                    move_dist = min(dist, self.GRID_SIZE * 0.5 * self.enemy_speed_multiplier)
                    enemy.move_to(enemy.pos + (direction / dist) * move_dist)

            # Resolve Interactions
            reward += self._resolve_interactions()
        
        self._update_animations()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.game_won:
                reward += self.REWARD_WIN
                self.score += self.REWARD_WIN
            else:
                reward += self.REWARD_LOSS
                self.score += self.REWARD_LOSS

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _resolve_interactions(self):
        reward = 0
        squads_to_remove = []
        for p_squad in self.player_squads:
            for e_squad in self.enemy_squads:
                if np.linalg.norm(p_squad.pos - e_squad.pos) < (p_squad.size + e_squad.size) / 2:
                    # Sound: Combat_Clash
                    self._create_particle_burst(p_squad.pos, 50, self.COLOR_ENEMY)
                    if self.np_random.random() < 0.7: # Player wins
                        squads_to_remove.append(e_squad)
                    else: # Enemy wins
                        squads_to_remove.append(p_squad)
                        reward += self.REWARD_LOSE_SQUAD
        
        self.player_squads = [s for s in self.player_squads if s not in squads_to_remove]
        self.enemy_squads = [s for s in self.enemy_squads if s not in squads_to_remove]
        if self.player_squads and self.selected_squad_idx >= len(self.player_squads):
            self.selected_squad_idx = 0

        # Citadel Capture
        for p_squad in self.player_squads:
            for citadel in self.citadels:
                if citadel.owner == 'enemy' and np.linalg.norm(p_squad.pos - citadel.pos) < (p_squad.size + citadel.size) / 2:
                    citadel.owner = 'player'
                    reward += self.REWARD_CAPTURE_CITADEL
                    self.resources += 5
                    self.captured_citadels_count += 1
                    # Sound: Citadel_Capture
                    self._create_particle_burst(citadel.pos, 100, self.COLOR_CITADEL_PLAYER)
                    
                    # Difficulty Scaling
                    if self.captured_citadels_count % 3 == 0:
                        self._add_enemy_squads(1)
                    if self.captured_citadels_count % 5 == 0:
                        self.enemy_speed_multiplier += 0.05
        return reward

    def _update_animations(self):
        lerp_factor = 0.2
        for squad in self.player_squads + self.enemy_squads:
            squad.update_visuals(lerp_factor)
        
        self.particles = [p for p in self.particles if p.update()]

    def _check_termination(self):
        if not self.player_squads:
            return True
        if all(c.owner == 'player' for c in self.citadels):
            self.game_won = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw Portals
        for i, portal in enumerate(self.portals):
            is_selected = self.game_mode == 'SELECT_PORTAL' and i == self.selected_portal_idx
            glow = 10 if is_selected else 5
            draw_glow_circle(self.screen, portal.pos, portal.size, self.COLOR_PORTAL, glow)

        # Draw Citadels
        for citadel in self.citadels:
            color = self.COLOR_CITADEL_PLAYER if citadel.owner == 'player' else self.COLOR_CITADEL_ENEMY
            base_rect = pygame.Rect(0, 0, citadel.size * 1.8, citadel.size * 0.7)
            base_rect.center = (citadel.pos[0], citadel.pos[1] + citadel.size * 0.2)
            pygame.draw.rect(self.screen, color, base_rect)
            roof_points = [
                (citadel.pos[0], citadel.pos[1] - citadel.size * 0.8),
                (citadel.pos[0] - citadel.size, citadel.pos[1] - citadel.size * 0.1),
                (citadel.pos[0] + citadel.size, citadel.pos[1] - citadel.size * 0.1)
            ]
            pygame.draw.polygon(self.screen, color, roof_points)

        # Draw Squad Trails
        for squad in self.player_squads + self.enemy_squads:
            color = self.COLOR_PLAYER if squad.is_player else self.COLOR_ENEMY
            if len(squad.trail) > 1:
                for i, pos in enumerate(squad.trail):
                    alpha = int(100 * (i / len(squad.trail)))
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(squad.size * 0.3), (*color, alpha))

        # Draw Squads
        for i, squad in enumerate(self.player_squads):
            self._draw_squad(squad, is_selected=(self.game_mode == 'SELECT_SQUAD' and i == self.selected_squad_idx))
        for squad in self.enemy_squads:
            self._draw_squad(squad)

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color = (*p.color, alpha)
            size = int(p.size * (p.lifespan / p.max_lifespan))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), size, color)

    def _draw_squad(self, squad, is_selected=False):
        color = self.COLOR_PLAYER if squad.is_player else self.COLOR_ENEMY
        pos = squad.visual_pos
        size = squad.size
        
        # Spear
        spear_end = (pos[0], pos[1] - size * 1.5)
        pygame.draw.line(self.screen, (150, 150, 150), (pos[0], pos[1]), spear_end, 2)
        pygame.draw.polygon(self.screen, (200, 200, 200), [spear_end, (spear_end[0]-3, spear_end[1]+8), (spear_end[0]+3, spear_end[1]+8)])
        
        # Shield/Body
        draw_glow_circle(self.screen, pos, size, color, 3)
        
        if is_selected:
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(size * 1.5), (255, 255, 255))

    def _render_ui(self):
        # Score and Resources
        text_surface = self.font_m.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 10, 10))
        
        text_surface = self.font_m.render(f"RESOURCES: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 10, 40))

        # Beat Bar
        beat_progress = self.beat_timer / self.BEAT_DURATION
        bar_width = self.WIDTH / 2
        bar_rect = pygame.Rect((self.WIDTH - bar_width) / 2, self.HEIGHT - 20, bar_width, 10)
        pygame.draw.rect(self.screen, self.COLOR_GRID, bar_rect, border_radius=5)
        fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, bar_width * beat_progress, bar_rect.height)
        pygame.draw.rect(self.screen, self.COLOR_PROMPT, fill_rect, border_radius=5)
        if self.beat_timer < 3: # Flash on beat
            pygame.draw.rect(self.screen, (255,255,255), bar_rect, 2, border_radius=5)

        # Action Prompt
        prompt = ""
        if self.game_mode == 'SELECT_SQUAD':
            prompt = "[ARROWS] Cycle Squad | [SPACE] Select Portal Target"
        elif self.game_mode == 'SELECT_PORTAL':
            prompt = "[ARROWS] Cycle Portal | [SPACE] Confirm Move | [SHIFT] Cancel"
        
        if self.game_over:
            msg = "VICTORY" if self.game_won else "DEFEAT"
            draw_text(self.screen, msg, (self.WIDTH / 2, self.HEIGHT / 2), self.font_l, (255,255,255), center=True)
        else:
            draw_text(self.screen, prompt, (self.WIDTH / 2, self.HEIGHT - 40), self.font_s, self.COLOR_PROMPT, center=True)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "resources": self.resources}

    def _create_particle_burst(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            size = self.np_random.integers(2, 5)
            self.particles.append(Particle(pos, vel, lifespan, color, size))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the environment directly for testing.
    # It will create a window and let you play the game with your keyboard.
    
    # Un-set the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping keys to MultiDiscrete actions
    # Action: [Movement, Space, Shift]
    # Movement: 0=none, 1=up, 2=down, 3=left, 4=right
    
    action = [0, 0, 0]
    clock = pygame.time.Clock()
    
    # Create a display window
    pygame.display.set_caption("Hoplite Portal Strategy")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

    while not done:
        # Process events to prevent the window from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        action = [0, 0, 0] # Reset action
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(30)

    env.close()
    pygame.quit()